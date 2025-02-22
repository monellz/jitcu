import tempfile
import random
import pytest
import torch

from jitcu import load_cuda_ops

@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_cpu_add_1d(ndim, dtype):
  code_str = r"""
#include "jitcu/tensor.h"
#include <cassert>
int64_t check_and_return_total_size(Tensor& c, const Tensor& a, const Tensor& b) {
  assert(a.ndim == b.ndim);
  assert(a.ndim == c.ndim);
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

extern "C" {

void add_i32(Tensor& c, const Tensor& a, const Tensor& b) {
  int64_t size = check_and_return_total_size(c, a, b);
  for (int i = 0; i < size; ++i) {
    c.data_ptr<int32_t>()[i] = a.data_ptr<int32_t>()[i] + b.data_ptr<int32_t>()[i];
  }
}

void add_f32(Tensor& c, const Tensor& a, const Tensor& b) {
  int64_t size = check_and_return_total_size(c, a, b);
  for (int i = 0; i < size; ++i) {
    c.data_ptr<float>()[i] = a.data_ptr<float>()[i] + b.data_ptr<float>()[i];
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
      func_names=["add_i32", "add_f32"],
      func_params=["t_t_t", "t_t_t"],
    )

    shape = [random.randint(1, 5) for _ in range(ndim)]

    a = torch.randint(0, 10, shape, dtype=dtype)
    b = torch.randint(0, 10, shape, dtype=dtype)
    c = torch.zeros_like(a)

    if dtype == torch.int32:
      lib.add_i32(c, a, b)
    elif dtype == torch.float32:
      lib.add_f32(c, a, b)
    else:
      raise ValueError(f"Unsupported dtype: {dtype}")
    assert torch.allclose(c, a + b)