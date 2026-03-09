#pragma once
#include <cstddef>

namespace miniDL {
namespace cuda {

void launch_adamw_step_float32(float* weight, const float* grad, float* m, float* v, float lr,
                               float beta1, float beta2, float eps, float weight_decay,
                               float beta1_t, float beta2_t, size_t N);

}  // namespace cuda
}  // namespace miniDL