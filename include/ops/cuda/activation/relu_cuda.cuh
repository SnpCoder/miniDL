#pragma once
#include <cstddef>

namespace miniDL {
namespace cuda {
// ==========================================
// GELU
// ==========================================
void launch_relu_forward_float32(const float* in, float* out, size_t total_elements);
void launch_relu_backward_float32(const float* in, const float* grad_out, float* grad_in,
                                  size_t total_elements);
}  // namespace cuda
}  // namespace miniDL