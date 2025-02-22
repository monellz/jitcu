#ifndef _JITCU_TENSOR_H
#define _JITCU_TENSOR_H

#include <cstdint>
#include <cstdio>

struct Tensor {
  void* data;
  int32_t ndim;
  int64_t* shape;
  int64_t* strides;

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