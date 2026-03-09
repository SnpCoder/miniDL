#include "../../../include/ops/loss/cross_entropy.h"

#include <algorithm>
#include <cmath>

#include "../../../include/ops/cuda/loss/cross_entropy_cuda.cuh"
#include "../../../include/utils/exception.h"

namespace miniDL {

std::vector<Tensor> CrossEntropyLossOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& pred   = inputs[0];  // 形状: [Batch, Classes] 也就是未经过 Softmax 的 Logits
    const Tensor& target = inputs[1];  // 形状: [Batch] 存储的是正确的类别索引 (0 ~ Classes-1)

    if (pred.ndim() != 2 || target.ndim() != 1) {
        MINIDL_THROW_INVALID_ARG(
            "CrossEntropyLoss requires pred to be 2D [N, C] and target to be 1D [N].");
    }

    size_t N = pred.shape()[0];
    size_t C = pred.shape()[1];

    Tensor out     = Tensor::empty(Shape({1}), pred.options());
    _softmax_probs = Tensor::empty(pred.shape(), pred.options());
    _target        = target;

    if (pred.device().isCpu()) {
        const float* p_ptr = pred.data_ptr<float>();
        const float* t_ptr = target.data_ptr<float>();
        float* out_ptr     = out.data_ptr<float>();
        float* prob_ptr    = _softmax_probs.data_ptr<float>();

        float total_loss = 0.0f;

        for (size_t i = 0; i < N; ++i) {
            size_t offset = i * C;

            // 1. Max Trick 防溢出
            float max_val = p_ptr[offset];
            for (size_t j = 1; j < C; ++j) { max_val = std::max(max_val, p_ptr[offset + j]); }

            // 2. 算 Exp 和 Sum
            float sum_exp = 0.0f;
            for (size_t j = 0; j < C; ++j) {
                float e              = std::exp(p_ptr[offset + j] - max_val);
                prob_ptr[offset + j] = e;
                sum_exp += e;
            }

            // 3. 归一化得出概率 P，并计算 Loss
            int class_idx = static_cast<int>(t_ptr[i]);
            for (size_t j = 0; j < C; ++j) {
                prob_ptr[offset + j] /= sum_exp;  // 完成 Softmax
            }

            // Loss = -log(P_{target})
            // 加 1e-8f 保护，绝对禁止出现 log(0) 导致 nan！
            float prob = prob_ptr[offset + class_idx];
            total_loss += -std::log(prob + 1e-8f);
        }

        // 求平均 Loss
        out_ptr[0] = total_loss / N;

    } else if (pred.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_cross_entropy_forward_float32(pred.data_ptr<float>(), target.data_ptr<float>(),
                                                   _softmax_probs.data_ptr<float>(),
                                                   out.data_ptr<float>(), N, C);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }

    return {out};
}

std::vector<Tensor> CrossEntropyLossOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];
    Tensor grad_pred       = Tensor::empty(_softmax_probs.shape(), _softmax_probs.options());
    Tensor grad_target;  // 目标标签不需要梯度

    size_t N = _softmax_probs.shape()[0];
    size_t C = _softmax_probs.shape()[1];

    if (grad_pred.device().isCpu()) {
        const float* prob_ptr = _softmax_probs.data_ptr<float>();
        const float* t_ptr    = _target.data_ptr<float>();
        const float* go_ptr   = grad_out.data_ptr<float>();
        float* gp_ptr         = grad_pred.data_ptr<float>();

        // 梯度的平均权重：dY / N
        float go_val = go_ptr[0] / N;

        for (size_t i = 0; i < N; ++i) {
            int class_idx = static_cast<int>(t_ptr[i]);
            size_t offset = i * C;

            for (size_t j = 0; j < C; ++j) {
                float p = prob_ptr[offset + j];
                // 核心求导公式：(P - Y) / N
                float y            = (j == class_idx) ? 1.0f : 0.0f;
                gp_ptr[offset + j] = (p - y) * go_val;
            }
        }
    } else if (grad_pred.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_cross_entropy_backward_float32(
            _softmax_probs.data_ptr<float>(), _target.data_ptr<float>(), grad_out.data_ptr<float>(),
            grad_pred.data_ptr<float>(), N, C);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }

    // 释放循环引用的缓存！打扫战场防内存泄漏！
    _softmax_probs = Tensor();
    _target        = Tensor();

    return {grad_pred, grad_target};
}

Tensor CrossEntropyLossOp::apply(const Tensor& pred, const Tensor& target) {
    auto op = std::make_shared<CrossEntropyLossOp>();
    return (*op)({pred, target})[0];
}

Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target) {
    return CrossEntropyLossOp::apply(pred, target);
}

}  // namespace miniDL