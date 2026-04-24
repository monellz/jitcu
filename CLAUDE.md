# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`jitcu` is a thin JIT layer for writing and iterating on CUDA kernels (and AscendC NPU kernels) from Python. It shells out to `nvcc` / `bisheng` at runtime, loads the resulting `.so` via `ctypes`, and calls into it with torch tensors. It is meant for debugging and perf tuning — not a production packaging system.

## Common commands

- Install for development: `uv sync --group dev` (creates `.venv/`, adds `pytest`, `pre-commit`, `ruff`, `ty`).
- Run a single test: `uv run pytest tests/test_cuda.py::test_gpu_add -v` (pass `-k` to narrow a parametrized case, e.g. `-k "ndim2 and float32"`).
- Run the CUDA / CPU / Ascend test files individually — they require different hardware (`cuda:0`, CPU, `npu:0`) and will fail elsewhere.
- Lint / format / type-check: `uv run pre-commit run --all-files` (ruff lint+format, ty, clang-format on C/C++/CUDA). Or directly: `uv run ruff check .`, `uv run ruff format .`, `uv run ty check`.
- Cluster-specific toolchain (spack-loaded `nvcc`, etc.) is not committed — set it up in your shell before running tests.

## Control env vars (read in `jitcu/env.py`)

- `JITCU_FORCE_RECOMPILE=1` — bypass the hash cache in `~/.cache/jitcu/cached_ops/<name>/`.
- `JITCU_VERBOSE=1` — add `-v` to the compiler invocation.
- `JITCU_NVCC_KEEP=1` — pass `--keep` to nvcc and also emit `<name>.sass` via `cuobjdump --dump-sass` for inspection.
- `JITCU_ENABLE_PROFILER=1` — define `JC_ENABLE_PROFILER` and force recompile; see profiler section below.
- Benchmarks (`jitcu/benchmark.py`) also read `verify=1` / `once=1`.

## Architecture

The pipeline is deliberately small. Understanding these four files covers most of it:

- `jitcu/core/` — backend package. `cuda.py::load_cuda_ops` and `ascend.py::load_ascend_ops` each accept either a path list or a raw source string (written to `<build_dir>/<name>.cu` or `.cpp`), build the compiler command, hash the sources + output `.so` to decide if recompile is needed, and return a `Library`. `common.py` holds the shared `JITCULogger` + `hash_files` helper. `externals.py` holds the third-party-library registry consumed by the `external_libs=` kwarg of `load_cuda_ops` (see below). The Ascend path has two sub-modes: device build via `bisheng` (`-x cce`, `--cce-aicore-arch=...`), or CPU simulation via a generated `CMakeLists.txt` that links `tikicpulib::<soc_version>`.
- `jitcu/library.py` — `Library` dlopens the `.so`, wires each exported symbol's `argtypes` from a `func_params` spec string, and wraps the call so Python-side torch tensors are converted to a C `Tensor` struct and the current CUDA/NPU stream is injected as arg 0. **ABI contract**: every exported function must be `extern "C"` with signature `void f(<stream_t>, Tensor&, ..., <scalars>)` — the stream is implicit in the Python call, not listed in `func_params`. Currently supported `func_params` codes: `t` (tensor pointer), `i32`, `i64`.
- `jitcu/data/include/jitcu/` — headers the user's kernel code is expected to `#include`. `tensor.h` defines the C `Tensor` struct and `DataType` enum that mirrors the Python side (`Tensor.Dtype.from_torch_dtype`). `utils.h` has `JITCU_CHECK` / `CUDA_CHECK` / `CUTLASS_CHECK` macros and cute-tensor dump helpers gated on `CUTE_HOST_DEVICE`. `dbg.h` is a vendored MIT `dbg(...)` macro — excluded from clang-format, don't reformat it. The include dir is auto-added to the compile command.
- `jitcu/profiler.py` + `jitcu/data/include/jitcu/profiler.h` — a flashinfer-derived device-side profiler. The kernel writes `(tag, globaltimer_lo)` pairs into a user-supplied `uint64_t*` buffer; the host decodes it to a perfetto trace via `tg4perfetto`. Tag layout is fixed (`BLOCK_GROUP_IDX_SHIFT=12`, `EVENT_IDX_SHIFT=2`, 3 event types), and the Python decoder in `profiler.py:decode_tag` must stay in sync with the device encoding in `profiler.h:encode_tag`. When `JC_ENABLE_PROFILER` is undefined the header provides no-op stubs so instrumented kernels still build.

### Adding a new scalar type to `func_params`

Touch both sides together: extend `Library.type_mapping` in `jitcu/library.py` and the wrapper conversion, and (if it's a new data type carried in tensors) keep `jitcu/data/include/jitcu/tensor.h::DataType` aligned with `library.py::Tensor.Dtype`.

### Adding a new entry to `external_libs`

`load_cuda_ops(external_libs=...)` accepts `list[str]` (all auto-search) or `dict[str, path|None]` (explicit path or `None` to auto-search). Search order per entry: user-supplied path → each `env_vars` env var → each `python_packages` (via `importlib.util.find_spec`) → `default_paths`. Candidates are accepted only if `<root>/<include_subdir>/<probe_header>` exists.

To register a new library, add an `ExternalLib(...)` entry to `EXTERNAL_LIBS` in `jitcu/core/externals.py`. `_resolve_link_lib` automatically rewrites `-l<name>` to `-l:lib<name>.so.<N>` when only the versioned SONAME is present (required for pip-installed NVIDIA packages that don't ship the unversioned symlink). rpath is added automatically for every resolved lib path.

### Build caching

The hash key is `md5(sources + existing .so)`. This means editing a `#include`d header in `jitcu/data/include/` does **not** invalidate the cache — rebuild with `JITCU_FORCE_RECOMPILE=1` when changing vendored headers.

## Gotchas

- First positional arg of every exported C function is the stream (`cudaStream_t` / `aclrtStream`) — but it is **not** listed in `func_params` and **not** passed from Python; the wrapper injects `torch.cuda.current_stream()` / `torch.npu.current_stream()`.
- `Tensor.from_torch_tensor` asserts `storage_offset() == 0`; sliced / offset tensors will trip it — `.contiguous()` first, or pass a base tensor plus offsets as scalars.
- `load_cuda_ops` rejects a `sources=` string that happens to be an existing file path — pass a list for files, a raw string for inline code.
- Ascend CPU mode generates a CMake project under `<build_dir>/<name>_cmake_build/` and requires `ASCEND_HOME_PATH` set and `tikicpulib` available at `$ASCEND_HOME_PATH/tools/tikicpulib/lib/cmake`.
- A warning fires if neither `extra_cflags` nor `extra_cuda_cflags` contains `-DNDEBUG` — intentional, CUTLASS/CUTE asserts are expensive.
- The build-cache hash covers source + `.so` only, not `external_libs` paths — if a resolver starts returning a different root, the embedded rpath in the old `.so` changes the file so the cache naturally invalidates, but env-var-driven flag changes without a new `.so` content diff will not. Use `JITCU_FORCE_RECOMPILE=1` if unsure.
- When using `external_libs=["nvshmem"]`, expose function names in the C source without colliding with NVSHMEM's own `nvshmem_*` symbols (e.g. don't declare your own `nvshmem_init`) — headers pull in the real declarations and gcc will error on conflict.
