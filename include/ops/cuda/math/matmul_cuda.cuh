#ifndef __MINIDL_OPERATOR_MATMUL_CUDA_H__
#define __MINIDL_OPERATOR_MATMUL_CUDA_H__

#include <cstddef>

namespace miniDL {
namespace cuda {
void launch_matmul_naive_float32(const float* a, const float* b, float* out, size_t M, size_t N,
                                 size_t K);

void launch_matmul_tiled_float32(const float* a, const float* b, float* out, size_t M, size_t N,
                                 size_t K);

void launch_matmul_vectorized_float32(const float* a, const float* b, float* out, size_t M,
                                      size_t N, size_t K);

void launch_matmul_auto_float32(const float* a, const float* b, float* out, size_t M, size_t N,
                                size_t K);
}  // namespace cuda
}  // namespace miniDL
#endif  // __MINIDL_OPERATOR_MATMUL_CUDA_H__