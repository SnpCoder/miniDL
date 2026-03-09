#include "../../../include/ops/activation/softmax.h"

#include <algorithm>
#include <cmath>

#ifdef USE_CUDA
#include "../../../include/ops/cuda/activation/softmax_cuda.cuh"
#endif

namespace miniDL {

std::vector<Tensor> SoftmaxOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& x = inputs[0];
    if (x.ndim() != 2) {
        // 现阶段，我们强制要求输入是 2D [Batch, Features]，按 Feature 维度做 Softmax
        throw std::runtime_error("Softmax currently only supports 2D tensors [Batch, Dim]");
    }

    size_t batch_size = x.shape()[0];
    size_t dim        = x.shape()[1];
    Tensor out        = Tensor::empty(x.shape(), x.options());

    if (x.device().isCpu()) {
        const float* x_ptr = x.data_ptr<float>();
        float* out_ptr     = out.data_ptr<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            size_t offset = b * dim;

            // 1. 寻找最大值 (极其重要的数值稳定性技巧！防止 e^x 溢出成 inf)
            float max_val = x_ptr[offset];
            for (size_t i = 1; i < dim; ++i) max_val = std::max(max_val, x_ptr[offset + i]);

            // 2. 计算 e^(x - max) 并求和
            float sum_exp = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                float e_val         = std::exp(x_ptr[offset + i] - max_val);
                out_ptr[offset + i] = e_val;
                sum_exp += e_val;
            }

            // 3. 归一化得出概率
            for (size_t i = 0; i < dim; ++i) { out_ptr[offset + i] /= sum_exp; }
        }
    } else if (x.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_softmax_forward_float32(x.data_ptr<float>(), out.data_ptr<float>(), batch_size,
                                             dim);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }

    // 【架构精髓】：我们需要把前向算出来的 Y(out) 保存下来，反向传播要用！
    _saved_out = out;
    return {out};
}

std::vector<Tensor> SoftmaxOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];
    Tensor grad_x          = Tensor::empty(grad_out.shape(), grad_out.options());

    size_t batch_size = grad_out.shape()[0];
    size_t dim        = grad_out.shape()[1];

    if (grad_out.device().isCpu()) {
        const float* y_ptr  = _saved_out.data_ptr<float>();  // 拿回前向保存的 Y
        const float* go_ptr = grad_out.data_ptr<float>();
        float* gx_ptr       = grad_x.data_ptr<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            size_t offset = b * dim;

            // 1. 计算 sum(dY * Y)
            float sum_gy = 0.0f;
            for (size_t i = 0; i < dim; ++i) { sum_gy += go_ptr[offset + i] * y_ptr[offset + i]; }

            // 2. 计算 dX = Y * (dY - sum_gy)
            for (size_t i = 0; i < dim; ++i) {
                gx_ptr[offset + i] = y_ptr[offset + i] * (go_ptr[offset + i] - sum_gy);
            }
        }
    } else if (grad_out.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_softmax_backward_float32(_saved_out.data_ptr<float>(),  // 前向缓存的 Y
                                              grad_out.data_ptr<float>(),    // 传进来的 dY
                                              grad_x.data_ptr<float>(),      // 算出来的 dX
                                              batch_size, dim);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return {grad_x};
}

}  // namespace miniDL