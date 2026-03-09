#pragma once
#include <cstddef>

namespace miniDL {
namespace cuda {

void launch_layernorm_forward_float32(const float* in, const float* weight, const float* bias,
                                      float* out, float* mean, float* rstd, size_t rows,
                                      size_t cols, float eps);

void launch_layernorm_backward_float32(const float* grad_out, const float* in, const float* weight,
                                       const float* mean, const float* rstd, float* grad_in,
                                       float* grad_weight, float* grad_bias, size_t rows,
                                       size_t cols, bool has_weight);
}  // namespace cuda
}  // namespace miniDL