#include <cuda_runtime.h>

#include "../../../../include/ops/cuda/activation/softmax_cuda.cuh"
#include "../../../../include/utils/exception.h"

namespace miniDL {
namespace cuda {

// 说明：工业级 Softmax 通常使用 Warp Reduce 优化，但为了保证框架可读性和通用性，
// 我们采用“一行一线程”策略，这在中小型维度下同样极具威力！
__global__ void softmax_forward_kernel(const float* __restrict__ in, float* __restrict__ out,
                                       size_t batch_size, size_t dim) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        size_t offset = b * dim;

        // 1. 找最大值防止指数溢出
        float max_val = in[offset];
        for (size_t i = 1; i < dim; ++i) {
            if (in[offset + i] > max_val) max_val = in[offset + i];
        }

        // 2. 计算 e^(x - max) 并求和
        float sum_exp = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float e_val     = expf(in[offset + i] - max_val);  // 使用 expf
            out[offset + i] = e_val;
            sum_exp += e_val;
        }

        // 3. 归一化
        for (size_t i = 0; i < dim; ++i) { out[offset + i] /= sum_exp; }
    }
}

__global__ void softmax_backward_kernel(const float* __restrict__ out_y,
                                        const float* __restrict__ grad_out,
                                        float* __restrict__ grad_in, size_t batch_size,
                                        size_t dim) {
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        size_t offset = b * dim;

        // 1. 计算 sum(dY * Y)
        float sum_gy = 0.0f;
        for (size_t i = 0; i < dim; ++i) { sum_gy += grad_out[offset + i] * out_y[offset + i]; }

        // 2. 计算 dX = Y * (dY - sum_gy)
        for (size_t i = 0; i < dim; ++i) {
            grad_in[offset + i] = out_y[offset + i] * (grad_out[offset + i] - sum_gy);
        }
    }
}

void launch_softmax_forward_float32(const float* in, float* out, size_t batch_size, size_t dim) {
    if (batch_size == 0) return;
    int threads = 256;
    int blocks  = (batch_size + threads - 1) / threads;  // 网格按照 Batch 分布
    softmax_forward_kernel<<<blocks, threads>>>(in, out, batch_size, dim);
}

void launch_softmax_backward_float32(const float* out_y, const float* grad_out, float* grad_in,
                                     size_t batch_size, size_t dim) {
    if (batch_size == 0) return;
    int threads = 256;
    int blocks  = (batch_size + threads - 1) / threads;
    softmax_backward_kernel<<<blocks, threads>>>(out_y, grad_out, grad_in, batch_size, dim);
}

}  // namespace cuda
}  // namespace miniDL