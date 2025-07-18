# Jitcu

A simple jit for quick cuda-related debugging and performance tuning

## Usage

See ```tests``` for examples.
* Now it supports Ascend devices, see ```tests/test_ascend_910a```

```python
from jitcu import load_cuda_ops

lib = load_cuda_ops(
    name="ops",
    sources=["source/path"],
    func_names=["func_1", "func_2"],
    # arg 0 of called function must be cudaStream_t, but we dont need to specify it there
    func_params=["t_t_t", "t_t_t_i32"], # only for input parameters, 't' means tensor
    arches=["90"],
    extra_cflags=[],
    extra_cuda_cflags=[],
    extra_ldflags=[],
    extra_include_paths=[],
    build_directory="./build",
)

# use it
a = torch.randn(2, device=0)
b = torch.randn(2, device=0)
c = torch.empty_like(a)
lib.func_1(c, a, b)
```
