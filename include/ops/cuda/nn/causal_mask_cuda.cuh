#pragma once
#include <cstddef>

namespace miniDL {
namespace cuda {

void launch_causal_mask_forward_float32(const float* in, float* out, size_t batch_heads,
                                        size_t seq_len);

void launch_causal_mask_backward_float32(const float* grad_out, float* grad_in, size_t batch_heads,
                                         size_t seq_len);

}  // namespace cuda
}  // namespace miniDL