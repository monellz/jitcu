import random

import pytest
import torch

from jitcu import load_ascend_ops


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("device", ["npu:0"])
def test_gpu_add(ndim, dtype, device):
    code_str = r"""
#include "jitcu/tensor.h"
#include "acl/acl.h"
#include "kernel_operator.h"
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
__global__ __aicore__ void _add_kernel(GM_ADDR c, GM_ADDR a, GM_ADDR b, int64_t size) {
  __gm__ T* c_ptr = reinterpret_cast<__gm__ T*>(c);
  __gm__ T* a_ptr = reinterpret_cast<__gm__ T*>(a);
  __gm__ T* b_ptr = reinterpret_cast<__gm__ T*>(b);

  int core_id = AscendC::GetBlockIdx();
  int core_num = AscendC::GetBlockNum();
  int block_size = (size + core_num - 1) / core_num;
  int curr_block_size = block_size < size - core_id * block_size ? block_size : size - core_id * block_size;

  AscendC::GlobalTensor<T> c_gm, a_gm, b_gm;

  a_gm.SetGlobalBuffer(a_ptr + core_id * block_size, curr_block_size);
  b_gm.SetGlobalBuffer(b_ptr + core_id * block_size, curr_block_size);
  c_gm.SetGlobalBuffer(c_ptr + core_id * block_size, curr_block_size);

  for (int i = 0; i < curr_block_size; ++i) {
    auto a_val = a_gm.GetValue(i);
    auto b_val = b_gm.GetValue(i);
    c_gm.SetValue(i, a_val + b_val);
  }
}

template<typename T>
void _add(Tensor& c, const Tensor& a, const Tensor& b, aclrtStream stream) {
  int64_t size = check_and_return_total_size(c, a, b);
  int core_num = 1;
  uint8_t* c_gm = reinterpret_cast<uint8_t*>(c.data_ptr());
  uint8_t* a_gm = reinterpret_cast<uint8_t*>(a.data_ptr());
  uint8_t* b_gm = reinterpret_cast<uint8_t*>(b.data_ptr());
  _add_kernel<T><<<core_num, nullptr, stream>>>(c_gm, a_gm, b_gm, size);
}

extern "C" {
void add(aclrtStream stream, Tensor& c, const Tensor& a, const Tensor& b) {
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
    lib = load_ascend_ops(
        name="add",
        sources=code_str,
        cce_aicore_arch="dav-c100",
        func_names=["add"],
        func_params=["t_t_t"],
        build_directory="./build",
    )

    shape = [random.randint(1, 5) for _ in range(ndim)]

    a = torch.randint(0, 32, shape, dtype=dtype, device=device)
    b = torch.randint(0, 32, shape, dtype=dtype, device=device)
    c = torch.zeros_like(a)

    lib.add(c, a, b)
    torch.npu.synchronize()
    print(f"{c=} {a=} {b=}", flush=True)
    torch.testing.assert_close(c, a + b)
