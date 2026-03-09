#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/shape/broadcast_cuda.cuh"
#include "../../../../include/utils/exception.h"

namespace miniDL {
namespace cuda {
// 物理广播：将 [N] 复制扩展为 [M, N]
__global__ void broadcast_1d_to_2d_kernel(const float* __restrict__ input,
                                          float* __restrict__ output, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        // 取模魔法：零内存开销实现广播寻址！
        output[idx] = input[idx % N];
    }
}

// 梯度缩减：将 [M, N] 的梯度沿 M 维求和，回退到 [N]
__global__ void reduce_sum_2d_to_1d_kernel(const float* __restrict__ grad_in,
                                           float* __restrict__ grad_out, size_t M, size_t N) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float sum = 0.0f;
        for (size_t i = 0; i < M; ++i) { sum += grad_in[i * N + col]; }
        grad_out[col] = sum;
    }
}

void launch_broadcast_1d_to_2d(const float* input, float* out, size_t M, size_t N) {
    if (M == 0 || N == 0) return;
    size_t total = M * N;
    int threads  = 256;
    int blocks   = (total + threads - 1) / threads;

    // 【修复】：只传入 4 个参数
    broadcast_1d_to_2d_kernel<<<blocks, threads>>>(input, out, M, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        MINIDL_THROW_RUNTIME("CUDA broadcast failed: %s", cudaGetErrorString(err));
}

void launch_reduce_sum_2d_to_1d(const float* grad_out, float* grad_b, size_t M, size_t N) {
    if (M == 0 || N == 0) return;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    reduce_sum_2d_to_1d_kernel<<<blocks, threads>>>(grad_out, grad_b, M, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        MINIDL_THROW_RUNTIME("CUDA reduce_sum failed: %s", cudaGetErrorString(err));
}
}  // namespace cuda
}  // namespace miniDL
