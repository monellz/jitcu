"""Multi-process NVSHMEM smoke test.

Bootstraps NVSHMEM with the UniqueID API (broadcast over torch.distributed /
gloo) and runs the classic `simple_shift` kernel — each rank `r` writes its id
to rank `(r+1) % N`, so rank `r` ends up reading the value from rank
`(r-1+N) % N`.

Launch modes
------------

Single-node (spawns `WORLD_SIZE` workers via `torch.multiprocessing.spawn`):
    uv run pytest tests/test_nvshmem.py -v

On SLURM, single node, N GPUs:
    srun --gres=gpu:2 uv run pytest tests/test_nvshmem.py -v

Multi-node via `torchrun` (e.g. 2 nodes, 1 GPU each → world_size=2). Pick one
node as rendezvous head; run on *every* node:
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=<0|1> \
             --rdzv_backend=c10d --rdzv_endpoint=<head_host>:29500 \
             tests/test_nvshmem.py

Multi-node via SLURM (2 nodes × 1 GPU). The GRES name (e.g. `gpu:1`,
`gpu:def:1`, `gpu:a100:1`) depends on how your cluster registered GPUs —
check `sinfo -N -o "%N %G"`:
    srun -p <partition> --nodes=2 --ntasks-per-node=1 --gres=gpu:1 \
         --time=00:10:00 \
         bash -c 'export RANK=$SLURM_PROCID
                  export WORLD_SIZE=$SLURM_NTASKS
                  export LOCAL_RANK=$SLURM_LOCALID
                  export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
                  export MASTER_PORT=29500
                  uv run python tests/test_nvshmem.py'

Under any env-driven launcher (`RANK` / `WORLD_SIZE` set), the pytest test
auto-detects it and runs a single rank instead of spawning — so
`torchrun ... -m pytest tests/test_nvshmem.py` also works.

If the SLURM command produces no output at all, typical causes are:
  * GRES type mismatch → job never got scheduled or got no GPUs.
  * `nvcc` / `spack load` not applied on compute nodes → first-time compile
    hangs or fails silently. Make sure the toolchain is visible to the
    shell spawned by `srun` (e.g. via `~/.bashrc` or explicit `spack load`
    inside the `bash -c '…'`).

For deeper NVSHMEM introspection (bootstrap transport, PE topology, IB/UCX
fabric selection, per-op traces) export any of the following before the
`python` invocation:
    NVSHMEM_DEBUG=INFO           # or WARN / TRACE — TRACE is very chatty
    NVSHMEM_DEBUG_SUBSYS=ALL     # or INIT,COLL,TRANSPORT,PROXY,…
    NVSHMEM_DEBUG_FILE=nvshmem.%h.%p.log   # %h=host, %p=pid; stderr if unset
    NVSHMEM_BOOTSTRAP=UID        # what this test uses; INFO lines show it
    NVSHMEM_REMOTE_TRANSPORT=ibrdma  # or ucx / ibdevx, if picking fails
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from jitcu import load_cuda_ops
from jitcu.core.externals import resolve_externals

NVSHMEM_SRC = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "jitcu/tensor.h"

using namespace jc;

namespace {
int32_t* g_dest = nullptr;
bool g_initialized = false;
}  // namespace

#define CHECK_NVSHMEM(call, name)                                      \
  do {                                                                 \
    int _r = (call);                                                   \
    if (_r != 0) {                                                     \
      fprintf(stderr, "[jitcu] %s failed: %d\n", name, _r);            \
      std::abort();                                                    \
    }                                                                  \
  } while (0)

__global__ void simple_shift_kernel(int32_t* destination) {
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;
  nvshmem_int_p(destination, mype, peer);
}

extern "C" {

void jc_nvshmem_get_uid(cudaStream_t /*stream*/, Tensor& uid_out) {
  nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
  CHECK_NVSHMEM(nvshmemx_get_uniqueid(&uid), "nvshmemx_get_uniqueid");
  std::memcpy(uid_out.data, &uid, sizeof(uid));
}

void jc_nvshmem_init(cudaStream_t /*stream*/, Tensor& uid_in, int32_t rank,
                  int32_t world_size) {
  nvshmemx_uniqueid_t uid;
  std::memcpy(&uid, uid_in.data, sizeof(uid));
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  CHECK_NVSHMEM(
      nvshmemx_set_attr_uniqueid_args(rank, world_size, &uid, &attr),
      "nvshmemx_set_attr_uniqueid_args");
  CHECK_NVSHMEM(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr),
                "nvshmemx_init_attr");
  g_dest = static_cast<int32_t*>(nvshmem_malloc(sizeof(int32_t)));
  if (!g_dest) {
    fprintf(stderr, "[jitcu] nvshmem_malloc failed\n");
    std::abort();
  }
  g_initialized = true;
}

void jc_nvshmem_run(cudaStream_t stream, Tensor& result) {
  simple_shift_kernel<<<1, 1, 0, stream>>>(g_dest);
  nvshmemx_barrier_all_on_stream(stream);
  cudaMemcpyAsync(result.data, g_dest, sizeof(int32_t),
                  cudaMemcpyDeviceToDevice, stream);
}

void jc_nvshmem_teardown(cudaStream_t /*stream*/) {
  if (g_dest) {
    nvshmem_free(g_dest);
    g_dest = nullptr;
  }
  if (g_initialized) {
    nvshmem_finalize();
    g_initialized = false;
  }
}

}  // extern "C"
"""


def _load_nvshmem_lib():
    return load_cuda_ops(
        name="nvshmem_simple_shift",
        sources=NVSHMEM_SRC,
        func_names=[
            "jc_nvshmem_get_uid",
            "jc_nvshmem_init",
            "jc_nvshmem_run",
            "jc_nvshmem_teardown",
        ],
        func_params=["t", "t_i32_i32", "t", ""],
        external_libs=["nvshmem"],
        extra_cuda_cflags=["-DNDEBUG"],
    )


def _log(rank: int, msg: str) -> None:
    host = os.uname().nodename
    print(f"[rank {rank} @ {host}] {msg}", flush=True)


def _worker(rank: int, world_size: int, local_rank: int | None = None):
    if local_rank is None:
        local_rank = rank
    _log(rank, f"start: world_size={world_size} local_rank={local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    _log(rank, "gloo init ok")

    lib = _load_nvshmem_lib()

    # rank 0 gets a UniqueID; broadcast it to all ranks over gloo (CPU tensor)
    uid = torch.zeros(128, dtype=torch.uint8)
    if rank == 0:
        lib.jc_nvshmem_get_uid(uid)
    dist.broadcast(uid, src=0)
    _log(rank, "uid broadcast ok")

    lib.jc_nvshmem_init(uid, rank, world_size)
    _log(rank, "nvshmem init ok")

    result = torch.zeros(1, dtype=torch.int32, device=f"cuda:{local_rank}")
    lib.jc_nvshmem_run(result)
    torch.cuda.synchronize()

    expected = (rank - 1 + world_size) % world_size
    got = int(result.item())
    assert got == expected, f"rank {rank}: got {got}, expected {expected}"
    _log(rank, f"simple_shift ok (got {got}, expected {expected})")

    lib.jc_nvshmem_teardown()
    dist.destroy_process_group()
    _log(rank, "done")


def _env_launched() -> bool:
    """True if an external launcher (torchrun, SLURM wrapper) already set
    RANK/WORLD_SIZE in the environment."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def test_nvshmem_simple_shift():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    try:
        resolve_externals(["nvshmem"])
    except RuntimeError as e:
        pytest.skip(f"nvshmem not available: {e}")

    if _env_launched():
        # Launched under torchrun / SLURM — this process is one rank.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.device_count() <= local_rank:
            pytest.skip(
                f"rank {rank}: local_rank {local_rank} but only "
                f"{torch.cuda.device_count()} CUDA devices visible"
            )
        _worker(rank, world_size, local_rank=local_rank)
        return

    # Single-node path: spawn N workers locally.
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"need at least {world_size} CUDA devices")

    # pre-compile in the parent so worker processes hit the build cache
    # and don't race on the .so path
    _load_nvshmem_lib()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    # Direct entry for multi-node launchers (torchrun, SLURM). RANK,
    # WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT must be set in the
    # environment — torchrun does this automatically.
    resolve_externals(["nvshmem"])
    _worker(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=int(os.environ.get("LOCAL_RANK", os.environ["RANK"])),
    )
