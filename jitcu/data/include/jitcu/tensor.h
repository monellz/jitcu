#ifndef _JITCU_TENSOR_H
#define _JITCU_TENSOR_H

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string_view>

namespace jc {

enum DataType {
  kUint64 = 0,
  kInt64 = 1,
  kFloat64 = 2,

  kUint32 = 3,
  kInt32 = 4,
  kFloat32 = 5,

  kUint16 = 6,
  kInt16 = 7,
  kFloat16 = 8,
  kBfloat16 = 9,

  kUint8 = 10,
  kInt8 = 11,

  kFloat8_e4m3fn = 12,
  kFloat8_e4m3fnuz = 13,
  kFloat8_e5m2 = 14,
  kFloat8_e5m2fnuz = 15,
};

constexpr std::string_view sv_of(DataType type) {
  switch (type) {
    case DataType::kUint64:
      return "u64";
    case DataType::kInt64:
      return "i64";
    case DataType::kFloat64:
      return "f64";
    case DataType::kUint32:
      return "u32";
    case DataType::kInt32:
      return "i32";
    case DataType::kFloat32:
      return "f32";
    case DataType::kUint16:
      return "u16";
    case DataType::kInt16:
      return "i16";
    case DataType::kFloat16:
      return "f16";
    case DataType::kBfloat16:
      return "bf16";
    case DataType::kUint8:
      return "u8";
    case DataType::kInt8:
      return "i8";
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

  template <typename T = void>
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
