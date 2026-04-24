"""Multi-process NVSHMEM smoke test.

Launches `WORLD_SIZE` workers via `torch.multiprocessing.spawn`, bootstraps
NVSHMEM with the UniqueID API (broadcast over torch.distributed / gloo), and
runs the classic `simple_shift` kernel — each rank `r` writes its id to rank
`(r+1) % N`, so rank `r` ends up reading the value from rank `(r-1+N) % N`.

Run directly:
    uv run pytest tests/test_nvshmem.py -v

On a SLURM cluster (single node, N GPUs):
    srun --gres=gpu:2 uv run pytest tests/test_nvshmem.py -v
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


def _worker(rank: int, world_size: int):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    lib = _load_nvshmem_lib()

    # rank 0 gets a UniqueID; broadcast it to all ranks over gloo (CPU tensor)
    uid = torch.zeros(128, dtype=torch.uint8)
    if rank == 0:
        lib.jc_nvshmem_get_uid(uid)
    dist.broadcast(uid, src=0)

    lib.jc_nvshmem_init(uid, rank, world_size)

    result = torch.zeros(1, dtype=torch.int32, device=f"cuda:{rank}")
    lib.jc_nvshmem_run(result)
    torch.cuda.synchronize()

    expected = (rank - 1 + world_size) % world_size
    got = int(result.item())
    assert got == expected, f"rank {rank}: got {got}, expected {expected}"

    lib.jc_nvshmem_teardown()
    dist.destroy_process_group()


def test_nvshmem_simple_shift():
    world_size = 2
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"need at least {world_size} CUDA devices")

    try:
        resolve_externals(["nvshmem"])
    except RuntimeError as e:
        pytest.skip(f"nvshmem not available: {e}")

    # pre-compile in the parent so worker processes hit the build cache
    # and don't race on the .so path
    _load_nvshmem_lib()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
