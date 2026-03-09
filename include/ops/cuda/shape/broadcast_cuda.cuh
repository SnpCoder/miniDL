#ifndef __MINIDL_OPERATOR_BROADCAST_CUDA_H__
#define __MINIDL_OPERATOR_BROADCAST_CUDA_H__

#include <cuda_runtime.h>

#include <cstddef>

namespace miniDL {
namespace cuda {

void launch_broadcast_1d_to_2d(const float* input, float* out, size_t M, size_t N);
void launch_reduce_sum_2d_to_1d(const float* grad_out, float* grad_b, size_t M, size_t N);
}  // namespace cuda
}  // namespace miniDL

#endif  // __MINIDL_OPERATOR_BROADCAST_CUDA_H__