import tempfile
import random
import pytest
import torch

from jitcu import load_cuda_ops

@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_gpu_add(ndim, dtype, device):
  code_str = r"""
#include "jitcu/tensor.h"
#include <cassert>
int64_t check_and_return_total_size(Tensor& c, const Tensor& a, const Tensor& b) {
  assert(a.ndim == b.ndim);
  assert(a.ndim == c.ndim);
  assert(a.dtype == b.dtype);
  assert(a.dtype == c.dtype);
  int64_t size = 1;
  for (int i = 0; i < a.ndim; ++i) {
    assert(a.size(i) == b.size(i));
    assert(a.size(i) == c.size(i));
    assert(a.stride(i) == b.stride(i));
    assert(a.stride(i) == c.stride(i));
    size *= a.size(i);
  }
  return size;
}

template<typename T>
__global__ void _add_kernel(T* c, const T* a, const T* b, int64_t size) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

template<typename T>
void _add(Tensor& c, const Tensor& a, const Tensor& b) {
  int64_t size = check_and_return_total_size(c, a, b);
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  _add_kernel<T><<<blocks, threads_per_block>>>(c.data_ptr<T>(), a.data_ptr<T>(), b.data_ptr<T>(), size);
}

extern "C" {
void add(Tensor& c, const Tensor& a, const Tensor& b) {
  if (c.dtype == kInt32) {
    _add<int32_t>(c, a, b);
  } else if (c.dtype == kFloat32) {
    _add<float>(c, a, b);
  } else {
    assert(false && "Unsupported dtype");
  }
}

}
  """
  with tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False) as f:
    f.write(code_str)
    f.flush()

    lib = load_cuda_ops(
      name="add",
      sources=[f.name],
      func_names=["add"],
      func_params=["t_t_t"],
    )

    shape = [random.randint(1, 5) for _ in range(ndim)]

    a = torch.randint(0, 10, shape, dtype=dtype, device=device)
    b = torch.randint(0, 10, shape, dtype=dtype, device=device)
    c = torch.zeros_like(a)

    lib.add(c, a, b)
    torch.cuda.synchronize()
    assert torch.allclose(c, a + b)