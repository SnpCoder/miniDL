#pragma once
#include <cstddef>

namespace miniDL {
namespace cuda {

void launch_cross_entropy_forward_float32(const float* logits, const float* targets, float* probs,
                                          float* out_loss, size_t N, size_t C);

void launch_cross_entropy_backward_float32(const float* probs, const float* targets,
                                           const float* grad_out, float* grad_pred, size_t N,
                                           size_t C);

}  // namespace cuda
}  // namespace miniDL