#include <cuda_runtime.h>

#include <algorithm>

#include "../../../../include/ops/cuda/math/matmul_cuda.cuh"
#include "../../../../include/utils/exception.h"

#ifdef ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

namespace miniDL {
namespace cuda {
__global__ void matmul_naive_float32(const float* __restrict__ A, const float* __restrict__ B,
                                     float* __restrict__ C, size_t M, size_t N, size_t K) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) { sum += A[row * K + k] * B[k * N + col]; }
        C[row * N + col] = sum;
    }
}

#define TILE_SIZE 32
__global__ void matmul_tiled_float32(const float* __restrict__ A, const float* __restrict__ B,
                                     float* __restrict__ C, size_t M, size_t N, size_t K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];  // avoid bank conflict

    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum  = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) { C[row * N + col] = sum; }
}

// K, N 必须都为4的倍数
#define TILE_M 4
#define TILE_N 4
__global__ void matmul_v4_vectorized_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B, float* __restrict__ C,
                                            size_t M, size_t N, size_t K) {
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE + 1];

    // 每个线程负责输出矩阵中的一个 4x4 小块
    int base_row = blockIdx.y * TILE_SIZE + threadIdx.y * TILE_M;
    int base_col = blockIdx.x * TILE_SIZE + threadIdx.x * TILE_N;

    float reg_c[TILE_M][TILE_N] = {0.0f};

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 使用 float4 并行读取 A 和 B
        for (int i = 0; i < TILE_M; ++i) {
            int a_row = base_row + i;
            int a_col = t * TILE_SIZE + threadIdx.x * TILE_N;  // threadIdx.x 控制列
            if (a_row < M && a_col + 3 < K) {
                float4 vec_a = *reinterpret_cast<const float4*>(&A[a_row * K + a_col]);
                s_a[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 0] = vec_a.x;
                s_a[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 1] = vec_a.y;
                s_a[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 2] = vec_a.z;
                s_a[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 3] = vec_a.w;
            } else {
                for (int v = 0; v < 4; ++v)
                    s_a[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + v] =
                        (a_row < M && a_col + v < K) ? A[a_row * K + a_col + v] : 0.0f;
            }

            int b_row = t * TILE_SIZE + threadIdx.y * TILE_M + i;
            int b_col = base_col;
            if (b_row < K && b_col + 3 < N) {
                float4 vec_b = *reinterpret_cast<const float4*>(&B[b_row * N + b_col]);
                s_b[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 0] = vec_b.x;
                s_b[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 1] = vec_b.y;
                s_b[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 2] = vec_b.z;
                s_b[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + 3] = vec_b.w;
            } else {
                for (int v = 0; v < 4; ++v)
                    s_b[threadIdx.y * TILE_M + i][threadIdx.x * TILE_N + v] =
                        (b_row < K && b_col + v < N) ? B[b_row * N + b_col + v] : 0.0f;
            }
        }
        __syncthreads();

        // 寄存器级重用计算
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

    // 将 4x4 的结果写回
    for (int i = 0; i < TILE_M; ++i) {
        for (int j = 0; j < TILE_N; ++j) {
            int row = base_row + i;
            int col = base_col + j;
            if (row < M && col < N) C[row * N + col] = reg_c[i][j];
        }
    }
}

void launch_matmul_naive_float32(const float* a, const float* b, float* out, size_t M, size_t N,
                                 size_t K) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    matmul_naive_float32<<<gridSize, blockSize>>>(a, b, out, M, N, K);
}

void launch_matmul_tiled_float32(const float* a, const float* b, float* out, size_t M, size_t N,
                                 size_t K) {
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_float32<<<gridSize, blockSize>>>(a, b, out, M, N, K);
}

void launch_matmul_vectorized_float32(const float* a, const float* b, float* out, size_t M,
                                      size_t N, size_t K) {
    // 因为每个线程算 4x4，所以 block 内的线程数是 (32/4) x (32/4) = 8x8 = 64
    // 个线程！极大降低了调度开销。
    dim3 block(TILE_SIZE / TILE_N, TILE_SIZE / TILE_M);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_v4_vectorized_kernel<<<grid, block>>>(a, b, out, M, N, K);
}

void launch_matmul_auto_float32(const float* a, const float* b, float* out, size_t M, size_t N,
                                size_t K) {
    if (M == 0 || N == 0 || K == 0) return;

#ifdef ENABLE_CUBLAS
    // 如果系统里装了 cuBLAS 且编译启用了，直接调用 Tensor Cores 大杀器！
    // (需注意 cuBLAS 默认是列优先，所以这里其实是算 B^T x A^T，为了简明我们这里省略具体调用代码)
    // cublas_matmul(...); return;
#endif

    int max_dim = std::max({M, N, K});

    // 1. 极小矩阵：直接用 V0 朴素版，启动快，不用分块同步
    if (max_dim <= 32) {
        launch_matmul_naive_float32(a, b, out, M, N, K);
    }
    // 2. 大矩阵，且内存严格对齐到 4 的倍数：启用 V4 性能怪兽
    else if (K % 4 == 0 && N % 4 == 0) {
        launch_matmul_vectorized_float32(a, b, out, M, N, K);
    }
    // 3. 中等矩阵，或内存不对齐（如 K=1023）：降级到 V1 共享内存版
    else {
        launch_matmul_tiled_float32(a, b, out, M, N, K);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("CUDA matmul auto-dispatcher failed: %s", cudaGetErrorString(err));
    }
}

}  // namespace cuda
}  // namespace miniDL