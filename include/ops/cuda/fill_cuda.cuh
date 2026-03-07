#ifndef __MINIDL_OPERATOR_FILL_CUDA_H__
#define __MINIDL_OPERATOR_FILL_CUDA_H__

#include <cstddef>

namespace miniDL {
namespace cuda {
void launch_fill_kernel_float32(float* data, float value, size_t n);
}  // namespace cuda
}  // namespace miniDL
#endif