# jitcu

A minimal JIT loader for CUDA kernels — write a kernel in a string or `.cu` file, call it from PyTorch a few lines later. Built for quick debugging and performance tuning, not for production packaging.

## Features

- **One-call JIT.** Pass source (string or file), function names, and an ABI spec; get back a callable object whose methods take torch tensors.
- **Content-hash cache.** Sources are hashed against the last built `.so` at `~/.cache/jitcu/cached_ops/<name>/` — unchanged sources skip recompilation.
- **CUDA Graph friendly.** The generated wrappers pass the current stream through, so captured graphs replay correctly (see `tests/test_cuda.py`).
- **Optional device-side profiler.** Flashinfer-derived lightweight instrumentation that exports to a Perfetto trace via `tg4perfetto`.
- **Drop-in helper headers.** `jitcu/tensor.h`, `jitcu/utils.h` (`JITCU_CHECK`, `CUDA_CHECK`, cute dump helpers), `jitcu/dbg.h`, `jitcu/profiler.h` — the include path is wired up automatically.

## Requirements

- Python ≥ 3.10, PyTorch, NumPy
- CUDA toolkit — `nvcc`, optionally `cuobjdump` for SASS dumps

## Installation

```bash
pip install -e .
# with the Perfetto trace exporter (tg4perfetto + protobuf)
pip install -e ".[profiler]"
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                    # runtime deps only
uv sync --extra profiler   # + Perfetto exporter
uv sync --group dev        # + pytest, pre-commit, ruff, ty
```

## Quick start

```python
import torch
from jitcu import load_cuda_ops

src = r"""
#include "jitcu/tensor.h"
using namespace jc;

template <typename T>
__global__ void _add(T* c, const T* a, const T* b, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

extern "C" void add(cudaStream_t stream, Tensor& c, const Tensor& a, const Tensor& b) {
  int64_t n = 1;
  for (int i = 0; i < a.ndim; ++i) n *= a.size(i);
  int threads = 256, blocks = (n + threads - 1) / threads;
  _add<float><<<blocks, threads, 0, stream>>>(
      c.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), n);
}
"""

lib = load_cuda_ops(
    name="ops",
    sources=src,                 # raw string OR list of file paths
    func_names=["add"],
    func_params=["t_t_t"],       # per-function ABI; see below
    arches=["90"],               # e.g. H100
)

a = torch.randn(1024, device="cuda")
b = torch.randn_like(a)
c = torch.empty_like(a)
lib.add(c, a, b)                 # stream is injected automatically
```

More examples: `tests/test_cuda.py`, `tests/test_cpu.py`.

### ABI contract

Every exported function must be `extern "C"` with this shape:

```cpp
void fn(cudaStream_t, Tensor& ..., <scalars>);
```

- Arg 0 is the CUDA stream. **It is injected by the wrapper from the current torch stream — do not list it in `func_params` and do not pass it from Python.**
- Remaining arguments are described by `func_params` using underscore-separated codes: `t` (`Tensor*`), `i32`, `i64`. Example: `"t_t_t_i32"` = three tensors and one `int32_t`.
- `Tensor` is the plain C struct in `jitcu/tensor.h`. Dtype codes line up with `Tensor.Dtype` on the Python side (see `jitcu/library.py`).

## Environment variables

| Variable | Effect |
| --- | --- |
| `JITCU_FORCE_RECOMPILE=1` | Ignore the hash cache and rebuild. |
| `JITCU_VERBOSE=1` | Pass `-v` to the compiler. |
| `JITCU_NVCC_KEEP=1` | Pass `--keep` to nvcc and emit `<name>.sass` via `cuobjdump --dump-sass`. |
| `JITCU_ENABLE_PROFILER=1` | Define `JC_ENABLE_PROFILER` (enables the device profiler) and force a rebuild. |

Cache root: `~/.cache/jitcu/` (see `jitcu/env.py`). The hash key is `md5(sources + existing .so)` — changes to bundled headers under `jitcu/data/include/` do **not** invalidate it; use `JITCU_FORCE_RECOMPILE=1`.

## Profiler

The profiler is optional — install with `pip install -e ".[profiler]"` to pull in `tg4perfetto` + `protobuf`.

With `JITCU_ENABLE_PROFILER=1`, include `jitcu/profiler.h` and instrument hot paths with `profiler::context_init / event_start / event_end / event_instant`. Pass a `uint64_t*` buffer from the host, then convert it to a Perfetto trace:

```python
from jitcu.profiler import export_to_perfetto_trace
export_to_perfetto_trace(buf, event_names=["load", "mma", "store"], file_name="trace.perfetto")
```

Open the resulting file at <https://ui.perfetto.dev/>.

## Benchmark helper

`jitcu.benchmark.benchmark(tag, fn, fn_ref, ...)` wraps `triton.testing.do_bench_cudagraph` and prints ms / TFLOPS/s / GB/s. Set `verify=1` to compare against `fn_ref`; set `once=1` to skip the cudagraph loop.

## Development

```bash
uv sync --group dev              # ruff, ty, pre-commit, pytest
uv run pre-commit install        # ruff (lint+format), ty, clang-format
uv run pytest tests/             # hardware-specific tests require the matching device
uv run pytest tests/test_cuda.py::test_gpu_add -v
```

Contributions are welcome. Please run `uv run pre-commit run --all-files` before opening a PR.

## Layout

```
jitcu/
  core/            # backends — cuda.py (load_cuda_ops), ascend.py (load_ascend_ops)
  library.py       # ctypes glue; torch.Tensor → jc::Tensor conversion
  profiler.py      # Perfetto trace export
  benchmark.py     # cudagraph-based benchmark helper
  env.py           # cache dirs + env var flags
  data/include/jitcu/
    tensor.h       # Tensor struct + DataType enum
    utils.h        # JITCU_CHECK / CUDA_CHECK / cute dump helpers
    profiler.h     # device-side profiler primitives
    dbg.h          # vendored dbg(...) macro
    all.h
tests/             # CUDA / CPU examples
```

## License

MIT. See [LICENSE](LICENSE).
