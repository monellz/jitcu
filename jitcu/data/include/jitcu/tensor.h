#ifndef _JITCU_TENSOR_H
#define _JITCU_TENSOR_H

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string_view>

namespace jc {

enum DataType {
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

constexpr std::string_view sv_of(DataType type) {
  switch (type) {
    case DataType::kInt64:
      return "i64";
    case DataType::kFloat64:
      return "f64";
    case DataType::kInt32:
      return "i32";
    case DataType::kFloat32:
      return "f32";
    case DataType::kFloat16:
      return "f16";
    case DataType::kBfloat16:
      return "bf16";
    case DataType::kFloat8_e4m3fn:
      return "f8_e4m3fn";
    case DataType::kFloat8_e4m3fnuz:
      return "f8_e4m3fnuz";
    case DataType::kFloat8_e5m2:
      return "f8_e5m2";
    case DataType::kFloat8_e5m2fnuz:
      return "f8_e5m2fnuz";
    default:
      return "Unknown";
  }
}

struct Tensor {
  void* data;
  int32_t ndim;
  int64_t* shape;
  int64_t* strides;
  DataType dtype;

  template <typename T>
  T* data_ptr() const {
    return static_cast<T*>(data);
  }

  inline int64_t size(int32_t dim) const { return shape[dim]; }

  inline int64_t stride(int32_t dim) const { return strides[dim]; }

  // to support dbg(...)
  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(";
    os << "dtype=" << sv_of(tensor.dtype) << ", ";
    os << "ndim=" << tensor.ndim << ", ";
    os << "shape=[";
    for (int32_t i = 0; i < tensor.ndim; ++i) {
      os << tensor.shape[i];
      if (i < tensor.ndim - 1) {
        os << ", ";
      }
    }
    os << "], ";
    os << "strides=[";
    for (int32_t i = 0; i < tensor.ndim; ++i) {
      os << tensor.strides[i];
      if (i < tensor.ndim - 1) {
        os << ", ";
      }
    }
    os << "], ";
    os << "data=" << tensor.data << "])";
    return os;
  }
};

}  // namespace jc

#endif  // _JITCU_TENSOR_H
