#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/activation/relu_cuda.cuh"
#include "../../../../include/utils/exception.h"

namespace miniDL {
namespace cuda {
__global__ void relu_forward_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { out[idx] = in[idx] > 0.0f ? in[idx] : 0.0f; }
}

__global__ void relu_backward_kernel(const float* __restrict__ in,
                                     const float* __restrict__ grad_out,
                                     float* __restrict__ grad_in, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { grad_in[idx] = in[idx] > 0.0f ? grad_out[idx] : 0.0f; }
}

void launch_relu_forward_float32(const float* in, float* out, size_t N) {
    if (N == 0) return;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(in, out, N);
}

void launch_relu_backward_float32(const float* in, const float* grad_out, float* grad_in,
                                  size_t N) {
    if (N == 0) return;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(in, grad_out, grad_in, N);
}

}  // namespace cuda
}  // namespace miniDL