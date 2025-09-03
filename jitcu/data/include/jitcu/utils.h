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

// following functions needs cutlass/cute
#if defined(CUTE_HOST_DEVICE)

template <typename T>
_JITCU_DEVICE void dump_cute_matrix(const T& addr) {
  static_assert(decltype(cute::rank(addr))::value == 2, "only support 2D tensor");
  int rows = decltype(cute::size<0>(addr))::value;
  int cols = decltype(cute::size<1>(addr))::value;
  printf("Dumping %d x %d cute matrix\n", rows, cols);

  printf("%5s", "");
  for (int j = 0; j < cols; ++j) {
    printf("%8d ", j);
  }
  printf("\n");
  for (int i = 0; i < rows; ++i) {
    printf("%3d | ", i);
    for (int j = 0; j < cols; ++j) {
      printf("%8.4f ", double(addr(i, j)));
    }
    printf("\n");
  }
  printf("\n");
}

template <typename T>
_JITCU_DEVICE void dump_cute_vector(const T& addr) {
  static_assert(decltype(cute::rank(addr))::value == 1, "only support 1D tensor");
  int n = decltype(cute::size<0>(addr))::value;
  printf("Dumping %d size cute vector\n", n);
  for (int i = 0; i < n; ++i) {
    printf("%8.4f ", double(addr(i)));
  }
  printf("\n");
}

template <bool ColumnMajor, int BLOCK_M, int BLOCK_N, typename TiledMma, typename TensorC>
__forceinline__ __device__ void store_Cregs_into_smem(TiledMma& tiled_mma, TensorC& acc,
                                                      typename TensorC::value_type* smem_ptr) {
  using DType = typename TensorC::value_type;
  using SmemCopyAtom = cute::Copy_Atom<cute::UniversalCopy<DType>, DType>;
  auto smem_tiled_copy = cute::make_tiled_copy_C(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy = smem_tiled_copy.get_thread_slice(threadIdx.x);
  using SmemLayout = std::conditional_t<
      ColumnMajor,
      decltype(cute::make_layout(cute::make_shape(cute::Int<BLOCK_M>{}, cute::Int<BLOCK_N>{}),
                                 cute::GenColMajor{})),
      decltype(cute::make_layout(cute::make_shape(cute::Int<BLOCK_M>{}, cute::Int<BLOCK_N>{}),
                                 cute::GenRowMajor{}))>;
  cute::Tensor smem = cute::make_tensor(cute::make_smem_ptr(smem_ptr), SmemLayout{});
  cute::Tensor tOs = smem_thr_copy.partition_D(cute::as_position_independent_swizzle_tensor(smem));
  cute::copy(smem_tiled_copy, smem_thr_copy.retile_S(acc), tOs);
}

#endif

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
