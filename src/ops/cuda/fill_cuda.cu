#include <cuda_runtime.h>

#include "../../../include/ops/cuda/fill_cuda.cuh"
#include "../../../include/utils/exception.h"

namespace miniDL {
namespace cuda {
__global__ void fill_vectorized_float4_kernel(float* __restrict__ data, float value, size_t N) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 4 <= N) {
        float4 vec                               = make_float4(value, value, value, value);
        reinterpret_cast<float4*>(&data[idx])[0] = vec;
    } else {
        for (size_t i = idx; i < N && i < idx + 4; ++i) { data[i] = value; }
    }
}

void launch_fill_kernel_float32(float* data, float value, size_t n) {
    if (n == 0) return;
    int threads = 256;
    int blocks  = (n + (threads * 4) - 1) / (threads * 4);

    fill_vectorized_float4_kernel<<<blocks, threads>>>(data, value, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("CUDA fill kernel failed: %s", cudaGetErrorString(err));
    }
}
}  // namespace cuda
}  // namespace miniDL