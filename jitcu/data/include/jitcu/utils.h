#ifndef _JITCU_UTILS_H
#define _JITCU_UTILS_H

#include <sstream>

namespace jc::utils {

template <typename... Args>
std::string check_failed_msg(const char* cond_str, Args&&... args) {
  std::ostringstream oss;
  oss << cond_str << " CHECK FAILED ";
  ((oss << args), ...);
  return oss.str();
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

#endif  // _JITCU_UTILS_H
