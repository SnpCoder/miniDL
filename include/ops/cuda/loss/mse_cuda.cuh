#ifndef __MINIDL_OPERATOR_LOSS_MSE_CUDA_H__
#define __MINIDL_OPERATOR_LOSS_MSE_CUDA_H__
#include <cstddef>

namespace miniDL {
namespace cuda {

void launch_mse_forward_float32(const float* pred, const float* target, float* loss_out, size_t N);
void launch_mse_backward_float32(const float* pred, const float* target, const float* grad_out,
                                 float* grad_pred, size_t N);

}  // namespace cuda
}  // namespace miniDL

#endif  // __MINIDL_OPERATOR_LOSS_MSE_CUDA_H__