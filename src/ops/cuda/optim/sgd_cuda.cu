#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/optim/sgd_cuda.cuh"
#include "../../../../include/utils/exception.h"

namespace miniDL {
namespace cuda {

__global__ void sgd_update_kernel(float* __restrict__ param, const float* __restrict__ grad,
                                  float* __restrict__ momentum_buf, float lr, float momentum,
                                  float weight_decay, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = grad[idx];

        // 1. Weight Decay (L2 正则化)
        if (weight_decay != 0.0f) { g += weight_decay * param[idx]; }

        // 2. Momentum
        if (momentum != 0.0f) {
            float buf         = momentum_buf[idx] * momentum + g;
            momentum_buf[idx] = buf;
            g                 = buf;  // 标准动量更新
        }

        // 3. 原地应用更新 (In-place)
        param[idx] -= lr * g;
    }
}

void launch_sgd_update_float32(float* param, const float* grad, float* momentum_buf, float lr,
                               float momentum, float weight_decay, size_t total_elements) {
    if (total_elements == 0) return;
    int threads = 256;
    int blocks  = (total_elements + threads - 1) / threads;

    sgd_update_kernel<<<blocks, threads>>>(param, grad, momentum_buf, lr, momentum, weight_decay,
                                           total_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("CUDA SGD update failed: %s", cudaGetErrorString(err));
    }
}

}  // namespace cuda
}  // namespace miniDL