#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/shape/transpose_cuda.cuh"
#include "../../../../include/utils/exception.h"

namespace miniDL {
namespace cuda {

// TILE_DIM=32 完美契合 Warp 大小 (32个线程)
#define TILE_DIM 32

__global__ void transpose_shared_kernel(const float* __restrict__ idata, float* __restrict__ odata,
                                        int rows, int cols) {
    // 极其关键的魔法：+1 Padding!
    // 如果没有这个 +1，按列写入 Shared Memory 时会产生严重的 Bank Conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 1. 读取阶段的全局坐标 (以输入矩阵为视角)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 完美的合并访存读取 (Coalesced Read)
    if (x < cols && y < rows) { tile[threadIdx.y][threadIdx.x] = idata[y * cols + x]; }

    __syncthreads();

    // 2. 写入阶段的全局坐标 (以输出矩阵为视角，注意 blockIdx.x 和 y 互换了！)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 完美的合并访存写入 (Coalesced Write)
    // 从 Shared Memory 按列读，写入到 Global Memory 的行
    if (x < rows && y < cols) { odata[y * rows + x] = tile[threadIdx.x][threadIdx.y]; }
}

void launch_transpose_kernel_float32(const float* in, float* out, int rows, int cols) {
    if (rows == 0 || cols == 0) return;

    dim3 block(TILE_DIM, TILE_DIM);
    // Grid 维度是根据输入矩阵的 cols 和 rows 来的
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);

    transpose_shared_kernel<<<grid, block>>>(in, out, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("CUDA transpose kernel failed: %s", cudaGetErrorString(err));
    }
}

}  // namespace cuda
}  // namespace miniDL