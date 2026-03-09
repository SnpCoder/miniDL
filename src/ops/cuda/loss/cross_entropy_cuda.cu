#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/loss/cross_entropy_cuda.cuh"

namespace miniDL {
namespace cuda {

// ============================================================================
// 前向传播：并行算 Softmax 和 Negative Log Likelihood
// ============================================================================
__global__ void ce_forward_kernel(const float* __restrict__ logits,
                                  const float* __restrict__ targets, float* __restrict__ probs,
                                  float* __restrict__ out_loss, size_t N, size_t C) {
    int i = blockIdx.x;  // 每个 Block 负责一个样本 (Row)
    if (i >= N) return;
    int tid = threadIdx.x;

    // 1. 求当前行的最大值 (防止指数爆炸)
    float max_val = -1e20f;
    for (int j = tid; j < C; j += blockDim.x) { max_val = max(max_val, logits[i * C + j]); }
    __shared__ float s_max[256];
    s_max[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = max(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    max_val = s_max[0];

    // 2. 求 exp 之和
    float sum_exp = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        float e          = expf(logits[i * C + j] - max_val);
        probs[i * C + j] = e;  // 顺手把 e 存下来
        sum_exp += e;
    }
    __shared__ float s_sum[256];
    s_sum[tid] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    sum_exp = s_sum[0];

    // 3. 归一化得出真实概率，并由 0 号线程计算出 Loss 累加到全局
    for (int j = tid; j < C; j += blockDim.x) { probs[i * C + j] /= sum_exp; }
    __syncthreads();  // 确保概率都写完了

    if (tid == 0) {
        int target_idx = static_cast<int>(targets[i]);
        float p        = probs[i * C + target_idx];
        float row_loss = -logf(p + 1e-8f);

        // 核心：用原子加法把所有 Batch 的 Loss 累加并求平均
        atomicAdd(out_loss, row_loss / static_cast<float>(N));
    }
}

// ============================================================================
// 反向传播：(P - Y) / N
// ============================================================================
__global__ void ce_backward_kernel(const float* __restrict__ probs,
                                   const float* __restrict__ targets,
                                   const float* __restrict__ grad_out,
                                   float* __restrict__ grad_pred, size_t N, size_t C) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        size_t i = idx / C;  // 当前元素属于哪个 Batch
        size_t j = idx % C;  // 当前元素是哪个 Class

        int target_idx = static_cast<int>(targets[i]);
        float p        = probs[idx];
        float y        = (j == target_idx) ? 1.0f : 0.0f;  // One-hot 标签

        // 获取外部传来的全局 Loss 梯度 (通常是 1.0)
        float go = grad_out[0] / static_cast<float>(N);

        grad_pred[idx] = (p - y) * go;
    }
}

void launch_cross_entropy_forward_float32(const float* logits, const float* targets, float* probs,
                                          float* out_loss, size_t N, size_t C) {
    // 【重要】：在累加之前，必须用 cudaMemset 把 Loss 指针清零！
    cudaMemset(out_loss, 0, sizeof(float));
    ce_forward_kernel<<<N, 256>>>(logits, targets, probs, out_loss, N, C);
}

void launch_cross_entropy_backward_float32(const float* probs, const float* targets,
                                           const float* grad_out, float* grad_pred, size_t N,
                                           size_t C) {
    size_t total = N * C;
    int threads  = 256;
    int blocks   = (total + threads - 1) / threads;
    ce_backward_kernel<<<blocks, threads>>>(probs, targets, grad_out, grad_pred, N, C);
}

}  // namespace cuda
}  // namespace miniDL