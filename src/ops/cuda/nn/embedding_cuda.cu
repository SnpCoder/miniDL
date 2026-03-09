#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/nn/embedding_cuda.cuh"

namespace miniDL {
namespace cuda {

// ============================================================================
// 前向传播：并行查表 (Lookup)
// ============================================================================
__global__ void embedding_forward_kernel(const float* __restrict__ indices,
                                         const float* __restrict__ weight, float* __restrict__ out,
                                         size_t total_tokens, size_t embed_dim, size_t vocab_size) {
    // 一个线程负责搬运 Embedding 向量里的一个 float
    size_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = total_tokens * embed_dim;

    if (idx < total_elements) {
        size_t token_pos = idx / embed_dim;  // 当前处理的是第几个 Token
        size_t dim_pos   = idx % embed_dim;  // 当前处理的是该 Token 的第几个特征维度

        // 取出词表索引 (把 float 强转为 int)
        int word_id = static_cast<int>(indices[token_pos]);

        // 越界保护
        if (word_id >= 0 && word_id < vocab_size) {
            out[idx] = weight[word_id * embed_dim + dim_pos];
        } else {
            out[idx] = 0.0f;  // 如果传入了非法的 word_id，输出全 0
        }
    }
}

// ============================================================================
// 反向传播：散列累加 (Scatter Add)
// ============================================================================
__global__ void embedding_backward_kernel(const float* __restrict__ grad_out,
                                          const float* __restrict__ indices,
                                          float* __restrict__ grad_weight, size_t total_tokens,
                                          size_t embed_dim, size_t vocab_size) {
    size_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = total_tokens * embed_dim;

    if (idx < total_elements) {
        size_t token_pos = idx / embed_dim;
        size_t dim_pos   = idx % embed_dim;

        int word_id = static_cast<int>(indices[token_pos]);

        if (word_id >= 0 && word_id < vocab_size) {
            float g = grad_out[idx];
            // 【核心护城河】：如果一句话里出现了两个相同的词（比如 "the"），
            // 它们的梯度必须累加到词表的同一个位置！必须使用原子锁！
            atomicAdd(&grad_weight[word_id * embed_dim + dim_pos], g);
        }
    }
}

void launch_embedding_forward_float32(const float* indices, const float* weight, float* out,
                                      size_t total_tokens, size_t embed_dim, size_t vocab_size) {
    size_t total_elements = total_tokens * embed_dim;
    if (total_elements == 0) return;
    int threads = 256;
    int blocks  = (total_elements + threads - 1) / threads;
    embedding_forward_kernel<<<blocks, threads>>>(indices, weight, out, total_tokens, embed_dim,
                                                  vocab_size);
}

void launch_embedding_backward_float32(const float* grad_out, const float* indices,
                                       float* grad_weight, size_t total_tokens, size_t embed_dim,
                                       size_t vocab_size) {
    size_t total_elements = total_tokens * embed_dim;
    if (total_elements == 0) return;
    int threads = 256;
    int blocks  = (total_elements + threads - 1) / threads;
    embedding_backward_kernel<<<blocks, threads>>>(grad_out, indices, grad_weight, total_tokens,
                                                   embed_dim, vocab_size);
}

}  // namespace cuda
}  // namespace miniDL