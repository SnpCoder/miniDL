#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/memory/contiguous_cuda.cuh"

namespace miniDL {
namespace cuda {

__global__ void contiguous_kernel(const float* __restrict__ in, float* __restrict__ out,
                                  TensorMeta meta, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 【核心算法】：一维索引降维解码，算出物理偏移量
        size_t offset = 0;
        size_t curr   = idx;

        // 从后往前解码多维坐标，并结合步长计算出错乱的物理索引
        for (int i = meta.ndim - 1; i >= 0; --i) {
            int coord = curr % meta.shape[i];
            offset += coord * meta.strides[i];
            curr /= meta.shape[i];
        }

        // 把散落的数据放进连续的新内存中
        out[idx] = in[offset];
    }
}

void launch_contiguous_float32(const float* in, float* out, const TensorMeta& meta, size_t N) {
    if (N == 0) return;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    contiguous_kernel<<<blocks, threads>>>(in, out, meta, N);
}

}  // namespace cuda
}  // namespace miniDL