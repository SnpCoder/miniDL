#include <cuda_runtime.h>

#include <algorithm>

#include "../../../../include/ops/cuda/math/bmm_cuda.cuh"
#include "../../../../include/utils/exception.h"

#ifdef ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

namespace miniDL {
namespace cuda {

// ============================================================================
// V1: 朴素版 BMM (适用于极小矩阵)
// ============================================================================
__global__ void bmm_naive_float32(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, size_t batch, size_t M, size_t N,
                                  size_t K) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b   = blockIdx.z;  // 获取 Batch 维度

    if (b < batch && row < M && col < N) {
        // 计算指针偏移量
        const float* batch_A = A + b * M * K;
        const float* batch_B = B + b * K * N;
        float* batch_C       = C + b * M * N;

        float sum = 0.0f;
        for (int k = 0; k < K; ++k) { sum += batch_A[row * K + k] * batch_B[k * N + col]; }
        batch_C[row * N + col] = sum;
    }
}

// ============================================================================
// V2: 共享内存分块版 BMM (解决 Memory Bound)
// ============================================================================
#define TILE_SIZE 32
__global__ void bmm_tiled_float32(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, size_t batch, size_t M, size_t N,
                                  size_t K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];  // avoid bank conflict

    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;
    size_t b   = blockIdx.z;

    if (b >= batch) return;

    // 偏移到当前 batch 的起始位置！
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;
    float* batch_C       = C + b * M * N;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = batch_A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = batch_B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) { batch_C[row * N + col] = sum; }
}

// ============================================================================
// V3: Float4 向量化 + 寄存器重用 BMM (性能怪兽)
// ============================================================================
#define TILE_M 4
#define TILE_N 4
__global__ void bmm_v4_vectorized_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                         float* __restrict__ C, size_t batch, size_t M, size_t N,
                                         size_t K) {
    size_t b = blockIdx.z;
    if (b >= batch) return;

    // 偏移到当前 batch
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;
    float* batch_C       = C + b * M * N;

    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE + 1];

    int base_row = blockIdx.y * TILE_SIZE + threadIdx.y * TILE_M;
    int base_col = blockIdx.x * TILE_SIZE + threadIdx.x * TILE_N;

    float reg_c[TILE_M][TILE_N] = {0.0f};

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 使用 float4 并行读取
        for (int i = 0; i < TILE_M; ++i) {
            int a_row = base_row + i;
            int a_col = t * TILE_SIZE + threadIdx.x * TILE_N;
            if (a_row < M && a_col + 3 < K) {
                float4 vec_a = *reinterpret_cast<const float4*>(&batch_A[a_row * K + a_col]);
                s_a[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 0] = vec_a.x;
                s_a[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 1] = vec_a.y;
                s_a[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 2] = vec_a.z;
                s_a[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 3] = vec_a.w;
            } else {
                for (int v = 0; v < 4; ++v)
                    s_a[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + v] =
                        (a_row < M && a_col + v < K) ? batch_A[a_row * K + a_col + v] : 0.0f;
            }

            int b_row = t * TILE_SIZE + threadIdx.y * TILE_M + i;
            int b_col = base_col;
            if (b_row < K && b_col + 3 < N) {
                float4 vec_b = *reinterpret_cast<const float4*>(&batch_B[b_row * N + b_col]);
                s_b[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 0] = vec_b.x;
                s_b[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 1] = vec_b.y;
                s_b[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 2] = vec_b.z;
                s_b[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 3] = vec_b.w;
            } else {
                for (int v = 0; v < 4; ++v)
                    s_b[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + v] =
                        (b_row < K && b_col + v < N) ? batch_B[b_row * N + b_col + v] : 0.0f;
            }
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            for (int i = 0; i < TILE_M; ++i) {
                for (int j = 0; j < TILE_N; ++j) {
                    reg_c[i][j] +=
                        s_a[threadIdx.y * TILE_M + i][k] * s_b[k][threadIdx.x * TILE_N + j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TILE_M; ++i) {
        for (int j = 0; j < TILE_N; ++j) {
            int row = base_row + i;
            int col = base_col + j;
            if (row < M && col < N) batch_C[row * N + col] = reg_c[i][j];
        }
    }
}

// ============================================================================
// BMM 智能调度器
// ============================================================================
void launch_bmm_auto_float32(const float* a, const float* b, float* out, size_t batch, size_t M,
                             size_t N, size_t K) {
    if (batch == 0 || M == 0 || N == 0 || K == 0) return;

#ifdef ENABLE_CUBLAS
    // 如果启用了 cuBLAS，使用其自带的 StridedBatched 极速接口
    // cublasSgemmStridedBatched(...); return;
#endif

    int max_dim = std::max({M, N, K});

    if (max_dim <= 32) {
        dim3 blockSize(16, 16, 1);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y,
                      batch);
        bmm_naive_float32<<<gridSize, blockSize>>>(a, b, out, batch, M, N, K);
    } else if (K % 4 == 0 && N % 4 == 0) {
        dim3 block(TILE_SIZE / TILE_N, TILE_SIZE / TILE_M, 1);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch);
        bmm_v4_vectorized_kernel<<<grid, block>>>(a, b, out, batch, M, N, K);
    } else {
        dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);
        dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch);
        bmm_tiled_float32<<<gridSize, blockSize>>>(a, b, out, batch, M, N, K);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("CUDA BMM auto-dispatcher failed: %s", cudaGetErrorString(err));
    }
}

}  // namespace cuda
}  // namespace miniDL