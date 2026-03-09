#include "../../../include/ops/loss/mse.h"

#include "../../../include/utils/exception.h"

#ifdef USE_CUDA
#include "../../../include/ops/cuda/loss/mse_cuda.cuh"
#endif

namespace miniDL {

std::vector<Tensor> MseLossOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& pred   = inputs[0];
    const Tensor& target = inputs[1];

    if (pred.shape() != target.shape()) {
        MINIDL_THROW_INVALID_ARG("MseLossOp: pred and target must have the same shape.");
    }

    size_t N = pred.element_num();

    // 输出是一个标量 (1D Tensor, size 1)
    Tensor loss = Tensor::empty(Shape({1}), pred.options());

    if (pred.device().isCpu()) {
        const float* p_ptr = pred.data_ptr<float>();
        const float* t_ptr = target.data_ptr<float>();
        float* l_ptr       = loss.data_ptr<float>();

        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float diff = p_ptr[i] - t_ptr[i];
            sum += diff * diff;
        }
        l_ptr[0] = sum / static_cast<float>(N);
    } else if (pred.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_mse_forward_float32(pred.data_ptr<float>(), target.data_ptr<float>(),
                                         loss.data_ptr<float>(), N);
#endif
    }
    return {loss};
}

std::vector<Tensor> MseLossOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];  // loss 的梯度
    const Tensor& pred     = this->inputs()[0];
    const Tensor& target   = this->inputs()[1];

    size_t N         = pred.element_num();
    Tensor grad_pred = Tensor::empty(pred.shape(), pred.options());

    // 我们通常不需要对 target 求导（因为 target 是真值常量），所以只返回 grad_pred
    if (pred.device().isCpu()) {
        const float* p_ptr  = pred.data_ptr<float>();
        const float* t_ptr  = target.data_ptr<float>();
        const float* go_ptr = grad_out.data_ptr<float>();
        float* gp_ptr       = grad_pred.data_ptr<float>();

        for (size_t i = 0; i < N; ++i) {
            gp_ptr[i] = (2.0f / static_cast<float>(N)) * (p_ptr[i] - t_ptr[i]) * go_ptr[0];
        }
    } else if (pred.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_mse_backward_float32(pred.data_ptr<float>(), target.data_ptr<float>(),
                                          grad_out.data_ptr<float>(), grad_pred.data_ptr<float>(),
                                          N);
#endif
    }

    // Target 梯度为空 (Tensor())
    return {grad_pred, Tensor()};
}

Tensor MseLossOp::apply(const Tensor& pred, const Tensor& target) {
    auto op = std::make_shared<MseLossOp>();
    return (*op)({pred, target})[0];
}

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    return MseLossOp::apply(pred, target);
}

}  // namespace miniDL