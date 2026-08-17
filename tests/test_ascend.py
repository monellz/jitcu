import random

import pytest
import torch

from jitcu import load_ascend_ops


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("device", ["npu:0"])
def test_gpu_add(ndim, dtype, device):
    # Minimal vector add: one AIV core walks the flat buffer. Kernel params are
    # __gm__ uint8_t* (host passes plain uint8_t*, cast to the typed __gm__ pointer
    # inside). Needs only jitcu/tensor.h — no acl / AscendC headers.
    code_str = r"""
#include "jitcu/tensor.h"
using namespace jc;

template <typename T>
__global__ __vector__ void _add_kernel(__gm__ uint8_t* c_, __gm__ uint8_t* a_, __gm__ uint8_t* b_, int64_t n) {
  auto c = reinterpret_cast<__gm__ T*>(c_);
  auto a = reinterpret_cast<__gm__ T*>(a_);
  auto b = reinterpret_cast<__gm__ T*>(b_);
  for (int64_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

extern "C" void add(void* stream, Tensor& c, Tensor& a, Tensor& b) {
  int64_t n = 1;
  for (int i = 0; i < a.ndim; ++i) n *= a.size(i);
  auto cp = reinterpret_cast<uint8_t*>(c.data);
  auto ap = reinterpret_cast<uint8_t*>(a.data);
  auto bp = reinterpret_cast<uint8_t*>(b.data);
  if (c.dtype == kInt32)
    _add_kernel<int32_t><<<1, nullptr, stream>>>(cp, ap, bp, n);
  else if (c.dtype == kFloat32)
    _add_kernel<float><<<1, nullptr, stream>>>(cp, ap, bp, n);
}
"""
    lib = load_ascend_ops(
        name="add",
        sources=code_str,
        func_specs={"add": "t_t_t"},
        build_directory="./build",
    )

    shape = [random.randint(1, 5) for _ in range(ndim)]
    a = torch.randint(0, 32, shape, dtype=dtype, device=device)
    b = torch.randint(0, 32, shape, dtype=dtype, device=device)
    c = torch.zeros_like(a)

    lib.add(c, a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(c, a + b)
