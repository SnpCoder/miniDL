#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/nn/causal_mask_cuda.cuh"

namespace miniDL {
namespace cuda {

__global__ void causal_mask_forward_kernel(const float* __restrict__ in, float* __restrict__ out,
                                           size_t batch_heads, size_t seq_len) {
    size_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_heads * seq_len * seq_len;

    if (idx < total) {
        size_t rem = idx % (seq_len * seq_len);
        size_t row = rem / seq_len;
        size_t col = rem % seq_len;

        // 【核心魔法】：如果列大于行，说明是未来的词，直接置为负无穷！
        if (col > row) {
            out[idx] = -1e9f;
        } else {
            out[idx] = in[idx];
        }
    }
}

__global__ void causal_mask_backward_kernel(const float* __restrict__ grad_out,
                                            float* __restrict__ grad_in, size_t batch_heads,
                                            size_t seq_len) {
    size_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_heads * seq_len * seq_len;

    if (idx < total) {
        size_t rem = idx % (seq_len * seq_len);
        size_t row = rem / seq_len;
        size_t col = rem % seq_len;

        // 负无穷的地方没有参与前向传播，所以梯度为 0
        if (col > row) {
            grad_in[idx] = 0.0f;
        } else {
            grad_in[idx] = grad_out[idx];
        }
    }
}

void launch_causal_mask_forward_float32(const float* in, float* out, size_t batch_heads,
                                        size_t seq_len) {
    size_t total = batch_heads * seq_len * seq_len;
    int threads  = 256;
    int blocks   = (total + threads - 1) / threads;
    causal_mask_forward_kernel<<<blocks, threads>>>(in, out, batch_heads, seq_len);
}

void launch_causal_mask_backward_float32(const float* grad_out, float* grad_in, size_t batch_heads,
                                         size_t seq_len) {
    size_t total = batch_heads * seq_len * seq_len;
    int threads  = 256;
    int blocks   = (total + threads - 1) / threads;
    causal_mask_backward_kernel<<<blocks, threads>>>(grad_out, grad_in, batch_heads, seq_len);
}

}  // namespace cuda
}  // namespace miniDL