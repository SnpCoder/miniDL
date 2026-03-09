#include "../../../include/ops/shape/broadcast.h"

#include "../../../include/ops/cuda/shape/broadcast_cuda.cuh"
#include "../../../include/utils/exception.h"

namespace miniDL {

std::vector<Tensor> BroadcastToOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& x = inputs[0];

    // 专门处理 [N] 广播到 [M, N] 的逻辑
    if (x.ndim() == 1 && _target_shape.size() == 2 && x.shape()[0] == _target_shape[1]) {
        size_t M   = _target_shape[0];
        size_t N   = _target_shape[1];
        Tensor out = Tensor::empty(_target_shape, x.options());

        if (x.device().isCpu()) {
            const float* in_ptr = x.data_ptr<float>();
            float* out_ptr      = out.data_ptr<float>();
            // CPU 广播逻辑：通过取模实现物理内存展开
            for (size_t i = 0; i < M * N; ++i) out_ptr[i] = in_ptr[i % N];
        } else if (x.device().isCuda()) {
#ifdef USE_CUDA
            // 【修复】：只传入 x 和 out，没有 a 和 b！
            cuda::launch_broadcast_1d_to_2d(x.data_ptr<float>(), out.data_ptr<float>(), M, N);
#else
            MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
        }
        return {out};
    }
    MINIDL_THROW_RUNTIME("BroadcastToOp: Only 1D to 2D broadcasting is supported currently.");
}

std::vector<Tensor> BroadcastToOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out   = grad_outputs[0];    // 形状 [M, N]
    const Tensor& original_x = this->inputs()[0];  // 形状 [N]

    Tensor grad_x = Tensor::empty(original_x.shape(), original_x.options());

    size_t M = grad_out.shape()[0];
    size_t N = grad_out.shape()[1];

    if (grad_out.device().isCpu()) {
        const float* gout = grad_out.data_ptr<float>();
        float* gx         = grad_x.data_ptr<float>();
        for (size_t j = 0; j < N; ++j) gx[j] = 0.0f;
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) { gx[j] += gout[i * N + j]; }
        }
    } else if (grad_out.device().isCuda()) {
#ifdef USE_CUDA
        // 【修复】：补全反向传播的 CUDA 调用，完成沿 M 维度的缩减求和
        cuda::launch_reduce_sum_2d_to_1d(grad_out.data_ptr<float>(), grad_x.data_ptr<float>(), M,
                                         N);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return {grad_x};
}

Tensor BroadcastToOp::apply(const Tensor& x, const Shape& target_shape) {
    auto op = std::make_shared<BroadcastToOp>(target_shape);
    return (*op)({x})[0];
}

Tensor broadcast_to(const Tensor& x, const Shape& target_shape) {
    return BroadcastToOp::apply(x, target_shape);
}

}  // namespace miniDL