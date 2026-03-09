#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/optim/adamw_cuda.cuh"

namespace miniDL {
namespace cuda {

__global__ void adamw_kernel(float* __restrict__ w, const float* __restrict__ g,
                             float* __restrict__ m, float* __restrict__ v, float lr, float beta1,
                             float beta2, float eps, float weight_decay, float beta1_t,
                             float beta2_t, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float weight = w[idx];
        float grad   = g[idx];

        // 1. 解耦的权重衰减 (Weight Decay)
        weight = weight - lr * weight_decay * weight;

        // 2. 更新一阶和二阶动量
        float m_val = beta1 * m[idx] + (1.0f - beta1) * grad;
        float v_val = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        m[idx]      = m_val;
        v[idx]      = v_val;

        // 3. 偏差校正 (Bias Correction)
        float m_hat = m_val / (1.0f - beta1_t);
        float v_hat = v_val / (1.0f - beta2_t);

        // 4. 应用自适应梯度更新
        w[idx] = weight - lr * (m_hat / (sqrtf(v_hat) + eps));
    }
}

void launch_adamw_step_float32(float* weight, const float* grad, float* m, float* v, float lr,
                               float beta1, float beta2, float eps, float weight_decay,
                               float beta1_t, float beta2_t, size_t N) {
    if (N == 0) return;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    adamw_kernel<<<blocks, threads>>>(weight, grad, m, v, lr, beta1, beta2, eps, weight_decay,
                                      beta1_t, beta2_t, N);
}

}  // namespace cuda
}  // namespace miniDL