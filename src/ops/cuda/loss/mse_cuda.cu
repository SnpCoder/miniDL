#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/loss/mse_cuda.cuh"
#include "../../../../include/utils/exception.h"

namespace miniDL {
namespace cuda {

// 前向：计算 (pred - target)^2 / N，并累加到单个 loss 标量中
__global__ void mse_forward_kernel(const float* __restrict__ pred, const float* __restrict__ target,
                                   float* __restrict__ loss, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float diff = pred[idx] - target[idx];
        // 简单粗暴的原子加法累加 (教学级实现，未来可优化为 Tree Reduction)
        atomicAdd(loss, (diff * diff) / static_cast<float>(N));
    }
}

// 反向：根据求导公式 dL/d(pred) = (2/N) * (pred - target) * grad_out
__global__ void mse_backward_kernel(const float* __restrict__ pred,
                                    const float* __restrict__ target,
                                    const float* __restrict__ grad_out,
                                    float* __restrict__ grad_pred, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float g_out    = grad_out[0];  // loss 的梯度通常是一个标量
        grad_pred[idx] = (2.0f / static_cast<float>(N)) * (pred[idx] - target[idx]) * g_out;
    }
}

void launch_mse_forward_float32(const float* pred, const float* target, float* loss_out, size_t N) {
    if (N == 0) return;
    // 每次计算前，必须先把显存里的 loss 清零！
    cudaMemset(loss_out, 0, sizeof(float));
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    mse_forward_kernel<<<blocks, threads>>>(pred, target, loss_out, N);
}

void launch_mse_backward_float32(const float* pred, const float* target, const float* grad_out,
                                 float* grad_pred, size_t N) {
    if (N == 0) return;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    mse_backward_kernel<<<blocks, threads>>>(pred, target, grad_out, grad_pred, N);
}

}  // namespace cuda
}  // namespace miniDL