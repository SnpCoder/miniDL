#ifndef __MINIDL_OPERATOR_ADD_CUDA_H__
#define __MINIDL_OPERATOR_ADD_CUDA_H__

#include <cstddef>

namespace miniDL {
namespace cuda {
void launch_add_kernel_float32(const float* a, const float* b, float* out, size_t n);

void launch_add_scalar_kernel_float32(const float* a, float scalar, float* out, size_t n);
}  // namespace cuda
}  // namespace miniDL
#endif  // __MINIDL_OPERATOR_ADD_CUDA_H__