#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/norm/layernorm_cuda.cuh"

namespace miniDL {
namespace cuda {

__global__ void layernorm_forward_kernel(const float* __restrict__ in,
                                         const float* __restrict__ weight,
                                         const float* __restrict__ bias, float* __restrict__ out,
                                         float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                         size_t rows, size_t cols, float eps) {
    // 一个 Block 负责一行数据
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid        = threadIdx.x;
    const float* x = in + row * cols;
    float* y       = out + row * cols;

    // 1. 计算当前线程负责的元素的局部和
    float thread_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) { thread_sum += x[i]; }

    // 共享内存用于 Block 级别的并行归约
    __shared__ float shared_val[256];
    shared_val[tid] = thread_sum;
    __syncthreads();

    // 归约求均值
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_val[tid] += shared_val[tid + stride];
        __syncthreads();
    }
    float mean = shared_val[0] / cols;

    // 2. 计算方差
    float thread_var_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = x[i] - mean;
        thread_var_sum += diff * diff;
    }
    shared_val[tid] = thread_var_sum;
    __syncthreads();

    // 归约求方差
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_val[tid] += shared_val[tid + stride];
        __syncthreads();
    }
    float var  = shared_val[0] / cols;
    float rstd = rsqrtf(var + eps);  // rsqrtf 硬件级快速计算倒数平方根

    // 3. 写入缓存供反向传播使用 (仅 0 号线程写一次)
    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    // 4. 标准化并应用 weight 和 bias
    for (int i = tid; i < cols; i += blockDim.x) {
        float normalized = (x[i] - mean) * rstd;
        float w          = (weight != nullptr) ? weight[i] : 1.0f;
        float b          = (bias != nullptr) ? bias[i] : 0.0f;
        y[i]             = normalized * w + b;
    }
}

__global__ void layernorm_backward_kernel(
    const float* __restrict__ grad_out, const float* __restrict__ in,
    const float* __restrict__ weight, const float* __restrict__ mean,
    const float* __restrict__ rstd, float* __restrict__ grad_in, float* __restrict__ grad_weight,
    float* __restrict__ grad_bias, size_t rows, size_t cols, bool has_weight) {
    // 一个 Block 负责一行数据的求导
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid         = threadIdx.x;
    const float* dy = grad_out + row * cols;
    const float* x  = in + row * cols;
    float* dx       = grad_in + row * cols;

    float m  = mean[row];
    float rs = rstd[row];

    // 1. 极速计算局部 sum_dy_w 和 sum_dy_w_xhat
    float thread_sum1 = 0.0f;
    float thread_sum2 = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x) {
        float x_hat = (x[i] - m) * rs;
        float w     = has_weight ? weight[i] : 1.0f;
        float dy_w  = dy[i] * w;

        thread_sum1 += dy_w;
        thread_sum2 += dy_w * x_hat;
    }

    // Shared memory 归约求这行的总和
    __shared__ float shared_sum1[256];
    __shared__ float shared_sum2[256];
    shared_sum1[tid] = thread_sum1;
    shared_sum2[tid] = thread_sum2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum1[tid] += shared_sum1[tid + stride];
            shared_sum2[tid] += shared_sum2[tid + stride];
        }
        __syncthreads();
    }

    float sum_dy_w      = shared_sum1[0];
    float sum_dy_w_xhat = shared_sum2[0];
    float f_cols        = (float)cols;
    float factor        = rs / f_cols;

    // 2. 计算 dX，并通过 atomicAdd 累加 dWeight 和 dBias
    for (int i = tid; i < cols; i += blockDim.x) {
        float x_hat = (x[i] - m) * rs;
        float w     = has_weight ? weight[i] : 1.0f;
        float dy_w  = dy[i] * w;

        // 终极公式计算 dX
        dx[i] = factor * (f_cols * dy_w - sum_dy_w - x_hat * sum_dy_w_xhat);

        // 跨行累加参数梯度，必须使用硬件级原子锁！
        if (has_weight) {
            float d_w = dy[i] * x_hat;
            float d_b = dy[i];
            atomicAdd(&grad_weight[i], d_w);
            atomicAdd(&grad_bias[i], d_b);
        }
    }
}

void launch_layernorm_forward_float32(const float* in, const float* weight, const float* bias,
                                      float* out, float* mean, float* rstd, size_t rows,
                                      size_t cols, float eps) {
    // 启动内核：每一行分配一个拥有 256 线程的 Block
    layernorm_forward_kernel<<<rows, 256>>>(in, weight, bias, out, mean, rstd, rows, cols, eps);
}

// 工业界 LayerNorm backward 涉及极其复杂的正交梯度推导。
// 我们可以先实现一个强健的 CPU fallback 用于算子连通性验证！）
void launch_layernorm_backward_float32(const float* grad_out, const float* in, const float* weight,
                                       const float* mean, const float* rstd, float* grad_in,
                                       float* grad_weight, float* grad_bias, size_t rows,
                                       size_t cols, bool has_weight) {
    layernorm_backward_kernel<<<rows, 256>>>(grad_out, in, weight, mean, rstd, grad_in, grad_weight,
                                             grad_bias, rows, cols, has_weight);
}
}  // namespace cuda
}  // namespace miniDL