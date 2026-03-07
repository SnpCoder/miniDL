#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/math/add_cuda.cuh"
#include "../../../../include/utils/exception.h"

namespace miniDL {
namespace cuda {
__global__ void add_vectorized_float4_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B, float* __restrict__ C,
                                             size_t N) {
    // each thread is responsible for 4 floats
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 4 <= N) {
        float4 a = reinterpret_cast<const float4*>(&A[idx])[0];
        float4 b = reinterpret_cast<const float4*>(&B[idx])[0];

        float4 c = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        reinterpret_cast<float4*>(&C[idx])[0] = c;
    } else {
        // remain data less than 4
        for (size_t i = idx; i < N && i < idx + 4; ++i) { C[i] = A[i] + B[i]; }
    }
}

__global__ void add_scalar_vectorized_float4_kernel(const float* __restrict__ A, float scalar,
                                                    float* __restrict__ C, size_t N) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 4 <= N) {
        float4 a = reinterpret_cast<const float4*>(&A[idx])[0];

        float4 c = make_float4(a.x + scalar, a.y + scalar, a.z + scalar, a.w + scalar);
        reinterpret_cast<float4*>(&C[idx])[0] = c;
    } else {
        // remain data less than 4
        for (size_t i = idx; i < N && i < idx + 4; ++i) { C[i] = A[i] + scalar; }
    }
}

__global__ void add_inplace_vectorized_float4_kernel(float* __restrict__ A,
                                                     const float* __restrict__ B, size_t N) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 4 <= N) {
        float4 a = reinterpret_cast<float4*>(&A[idx])[0];
        float4 b = reinterpret_cast<const float4*>(&B[idx])[0];

        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
        a.w += b.w;
        reinterpret_cast<float4*>(&A[idx])[0] = a;
    } else {
        for (size_t i = idx; i < N && i < idx + 4; ++i) { A[i] += B[i]; }
    }
}

__global__ void add_scalar_inplace_vectorized_float4_kernel(float* __restrict__ A, float scalar,
                                                            size_t N) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 4 <= N) {
        float4 a = reinterpret_cast<float4*>(&A[idx])[0];
        a.x += scalar;
        a.y += scalar;
        a.z += scalar;
        a.w += scalar;
        reinterpret_cast<float4*>(&A[idx])[0] = a;
    } else {
        for (size_t i = idx; i < N && i < idx + 4; ++i) { A[i] += scalar; }
    }
}

// for host
void launch_add_kernel_float32(const float* a, const float* b, float* out, size_t n) {
    if (n == 0) return;
    int threads = 256;
    int blocks  = (n + (threads * 4) - 1) / (threads * 4);

    add_vectorized_float4_kernel<<<blocks, threads>>>(a, b, out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("CUDA add kernel failed: %s", cudaGetErrorString(err));
    }
}

void launch_add_scalar_kernel_float32(const float* a, float scalar, float* out, size_t n) {
    if (n == 0) return;
    int threads = 256;
    int blocks  = (n + (threads * 4) - 1) / (threads * 4);

    add_scalar_vectorized_float4_kernel<<<blocks, threads>>>(a, scalar, out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("CUDA add_scalar kernel failed: %s", cudaGetErrorString(err));
    }
}

void launch_add_inplace_kernel_float32(float* a, const float* b, size_t n) {
    if (n == 0) return;
    int threads = 256;
    int blocks  = (n + (threads * 4) - 1) / (threads * 4);

    add_inplace_vectorized_float4_kernel<<<blocks, threads>>>(a, b, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("CUDA add_inplace kernel failed: %s", cudaGetErrorString(err));
    }
}

void launch_add_scalar_inplace_kernel_float32(float* a, float scalar, size_t n) {
    if (n == 0) return;
    int threads = 256;
    int blocks  = (n + (threads * 4) - 1) / (threads * 4);

    add_scalar_inplace_vectorized_float4_kernel<<<blocks, threads>>>(a, scalar, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("CUDA add_scalar_inplace kernel failed: %s", cudaGetErrorString(err));
    }
}

}  // namespace cuda
}  // namespace miniDL