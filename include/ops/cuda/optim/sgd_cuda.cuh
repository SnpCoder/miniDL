#ifndef __MINIDL_OPERATOR_OPTIM_SGD_H__
#define __MINIDL_OPERATOR_OPTIM_SGD_H__
#include <cstddef>

namespace miniDL {
namespace cuda {

// 原地更新内核：param = param - lr * grad
void launch_sgd_update_float32(float* param, const float* grad, float* momentum_buf, float lr,
                               float momentum, float weight_decay, size_t total_elements);

}  // namespace cuda
}  // namespace miniDL
#endif  // __MINIDL_OPERATOR_OPTIM_SGD_H__