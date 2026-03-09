#pragma once
#include <cstddef>

namespace miniDL {
namespace cuda {
// ==========================================
// Softmax (2D: [Batch, Dim])
// ==========================================
void launch_softmax_forward_float32(const float* in, float* out, size_t batch_size, size_t dim);
void launch_softmax_backward_float32(const float* out_y, const float* grad_out, float* grad_in,
                                     size_t batch_size, size_t dim);

}  // namespace cuda
}  // namespace miniDL