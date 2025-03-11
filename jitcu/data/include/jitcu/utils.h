#ifndef _JITCU_UTILS_H
#define _JITCU_UTILS_H

#include <sstream>

#ifdef __CUDACC__
#define _JITCU_DEVICE __device__ __forceinline__
#else
#define _JITCU_DEVICE
#endif

namespace jc::utils {

template <typename... Args>
std::string check_failed_msg(const char* cond_str, Args&&... args) {
  std::ostringstream oss;
  oss << cond_str << " CHECK FAILED ";
  ((oss << args), ...);
  return oss.str();
}

template <typename T>
_JITCU_DEVICE void dump_rowmajor_matrix(T* addr, int rows, int cols) {
  printf("Dumping %d x %d rowmajor matrix\n", rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%8.4f ", double(addr[i * cols + j]));
    }
    printf("\n");
  }
  printf("\n");
}

template <typename T>
_JITCU_DEVICE void dump_cute_matrix(const T& addr, int rows, int cols) {
  printf("Dumping %d x %d cute matrix\n", rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%8.4f ", double(addr(i, j)));
    }
    printf("\n");
  }
  printf("\n");
}

}  // namespace jc::utils

#define JITCU_CHECK_MSG(cond, ...) ::jc::utils::check_failed_msg(#cond, __VA_ARGS__)

#define JITCU_CHECK(cond, ...)                                                                     \
  if (!(cond)) {                                                                                   \
    throw std::runtime_error(                                                                      \
        JITCU_CHECK_MSG(cond, "at ", __func__, ", ", __FILE__, ":", __LINE__, ", ", __VA_ARGS__)); \
  }

#define CUTLASS_CHECK(status)                                                       \
  {                                                                                 \
    cutlass::Status error = status;                                                 \
    JITCU_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

#define CUDA_CHECK(status)                                        \
  {                                                               \
    cudaError_t error = status;                                   \
    JITCU_CHECK(error == cudaSuccess, cudaGetErrorString(error)); \
  }

#define CUDA_CHECK_KERNEL_LAUNCH() CUDA_CHECK(cudaGetLastError())

#define CUTE_PRINT(x)    \
  do {                   \
    ::cute::print(#x);   \
    ::cute::print(": "); \
    ::cute::print(x);    \
    ::cute::print("\n"); \
  } while (0)

#endif  // _JITCU_UTILS_H
