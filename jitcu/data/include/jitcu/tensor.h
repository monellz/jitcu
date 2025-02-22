#ifndef _JITCU_TENSOR_H
#define _JITCU_TENSOR_H

#include <cstdint>
#include <cstdio>

enum DataType{
  kInt64 = 0,
  kFloat64 = 1,
  kInt32 = 2,
  kFloat32 = 3,
  kFloat16 = 4,
  kBfloat16 = 5,
  kFloat8_e4m3fn = 6,
  kFloat8_e4m3fnuz = 7,
  kFloat8_e5m2 = 8,
  kFloat8_e5m2fnuz = 9,
};

struct Tensor {
  void* data;
  int32_t ndim;
  int64_t* shape;
  int64_t* strides;
  DataType dtype;

  template<typename T>
  T* data_ptr() const {
    return static_cast<T*>(data);
  }

  inline int64_t size(int32_t dim) const {
    return shape[dim];
  }

  inline int64_t stride(int32_t dim) const {
    return strides[dim];
  }
};

#endif // _JITCU_TENSOR_H