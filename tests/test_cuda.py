import random
import tempfile

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
__global__ void _add_kernel(T* c, const T* a, const T* b, int64_t size) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

template<typename T>
void _add(Tensor& c, const Tensor& a, const Tensor& b, cudaStream_t stream) {
  int64_t size = check_and_return_total_size(c, a, b);
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  _add_kernel<T><<<blocks, threads_per_block, 0, stream>>>(c.data_ptr<T>(), a.data_ptr<T>(), b.data_ptr<T>(), size);
}

extern "C" {
void add(cudaStream_t stream, Tensor& c, const Tensor& a, const Tensor& b) {
  if (c.dtype == kInt32) {
    _add<int32_t>(c, a, b, stream);
  } else if (c.dtype == kFloat32) {
    _add<float>(c, a, b, stream);
  } else {
    assert(false && "Unsupported dtype");
  }
}

}
  """
    lib = load_cuda_ops(
        name="add",
        sources=code_str,
        func_names=["add"],
        func_params=["t_t_t"],
    )

    shape = [random.randint(1, 5) for _ in range(ndim)]

    a = torch.randint(0, 10, shape, dtype=dtype, device=device)
    b = torch.randint(0, 10, shape, dtype=dtype, device=device)
    c = torch.zeros_like(a)

    lib.add(c, a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b)

    # test it can be captured by cuda graph
    with torch.cuda.stream(torch.cuda.Stream()):
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.current_stream()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            lib.add(c, a, b)
        torch.cuda.synchronize()
        c.fill_(0)
        torch.testing.assert_close(c, torch.zeros_like(c))
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(c, a + b)
