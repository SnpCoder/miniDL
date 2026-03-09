#pragma once
#include <cstddef>

namespace miniDL {
namespace cuda {

void launch_embedding_forward_float32(const float* indices, const float* weight, float* out,
                                      size_t total_tokens, size_t embed_dim, size_t vocab_size);

void launch_embedding_backward_float32(const float* grad_out, const float* indices,
                                       float* grad_weight, size_t total_tokens, size_t embed_dim,
                                       size_t vocab_size);

}  // namespace cuda
}  // namespace miniDL