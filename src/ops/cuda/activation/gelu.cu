#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/activation/gelu_cuda.cuh"
#include "../../../../include/utils/exception.h"

namespace miniDL {
namespace cuda {

// 常量预定义 (GELU)
#define SQRT_2_OVER_PI 0.7978845608f
#define COEF 0.044715f

__global__ void gelu_forward_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x     = in[idx];
        float inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
        out[idx]    = 0.5f * x * (1.0f + tanhf(inner));  // 注意使用 tanhf 针对 float 优化
    }
}

__global__ void gelu_backward_kernel(const float* __restrict__ in,
                                     const float* __restrict__ grad_out,
                                     float* __restrict__ grad_in, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x     = in[idx];
        float inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
        float t     = tanhf(inner);

        float left  = 0.5f * (1.0f + t);
        float sech2 = 1.0f - t * t;
        float right = 0.5f * x * sech2 * SQRT_2_OVER_PI * (1.0f + 3.0f * COEF * x * x);

        grad_in[idx] = grad_out[idx] * (left + right);
    }
}

void launch_gelu_forward_float32(const float* in, float* out, size_t N) {
    if (N == 0) return;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    gelu_forward_kernel<<<blocks, threads>>>(in, out, N);
}

void launch_gelu_backward_float32(const float* in, const float* grad_out, float* grad_in,
                                  size_t N) {
    if (N == 0) return;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    gelu_backward_kernel<<<blocks, threads>>>(in, grad_out, grad_in, N);
}

}  // namespace cuda
}  // namespace miniDL