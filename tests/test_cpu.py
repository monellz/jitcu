import random
import tempfile

import pytest
import torch

from jitcu import load_cuda_ops


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_cpu_add(ndim, dtype):
    code_str = r"""
#include "jitcu/tensor.h"
#include <cassert>
using namespace jc;
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
void _add(Tensor& c, const Tensor& a, const Tensor& b) {
  int64_t size = check_and_return_total_size(c, a, b);
  for (int i = 0; i < size; ++i) {
    c.data_ptr<T>()[i] = a.data_ptr<T>()[i] + b.data_ptr<T>()[i];
  }
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

        a = torch.randint(0, 10, shape, dtype=dtype)
        b = torch.randint(0, 10, shape, dtype=dtype)
        c = torch.zeros_like(a)

        lib.add(c, a, b)
        assert torch.allclose(c, a + b)
