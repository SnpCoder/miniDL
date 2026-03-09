#include "../../../include/ops/norm/layernorm.h"

#include <cmath>

#include "../../../include/utils/exception.h"

#ifdef USE_CUDA
#include "../../../include/ops/cuda/norm/layernorm_cuda.cuh"
#endif

namespace miniDL {

std::vector<Tensor> LayerNormOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& x      = inputs[0];
    const Tensor& weight = inputs[1];  // 可以是空的 (undefined)
    const Tensor& bias   = inputs[2];  // 可以是空的 (undefined)

    // 扁平化视角：前面的维度视为 Rows，最后一个(或几个)维度视为 Cols
    size_t cols = _normalized_shape.elements();
    size_t rows = x.element_num() / cols;

    Tensor out  = Tensor::empty(x.shape(), x.options());
    _saved_mean = Tensor::empty(Shape({rows, 1}), x.options());
    _saved_rstd = Tensor::empty(Shape({rows, 1}), x.options());

    if (x.device().isCpu()) {
        const float* x_ptr = x.data_ptr<float>();
        const float* w_ptr = weight.defined() ? weight.data_ptr<float>() : nullptr;
        const float* b_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
        float* y_ptr       = out.data_ptr<float>();
        float* m_ptr       = _saved_mean.data_ptr<float>();
        float* r_ptr       = _saved_rstd.data_ptr<float>();

        for (size_t r = 0; r < rows; ++r) {
            const float* curr_x = x_ptr + r * cols;
            float* curr_y       = y_ptr + r * cols;

            // 1. Mean
            float sum = 0.0f;
            for (size_t c = 0; c < cols; ++c) sum += curr_x[c];
            float mean = sum / cols;
            m_ptr[r]   = mean;

            // 2. Variance & Rstd
            float var_sum = 0.0f;
            for (size_t c = 0; c < cols; ++c) {
                float diff = curr_x[c] - mean;
                var_sum += diff * diff;
            }
            float var  = var_sum / cols;
            float rstd = 1.0f / std::sqrt(var + _eps);
            r_ptr[r]   = rstd;

            // 3. Normalize & Scale/Shift
            for (size_t c = 0; c < cols; ++c) {
                float norm_val = (curr_x[c] - mean) * rstd;
                float w        = w_ptr ? w_ptr[c] : 1.0f;
                float b        = b_ptr ? b_ptr[c] : 0.0f;
                curr_y[c]      = norm_val * w + b;
            }
        }
    } else if (x.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_layernorm_forward_float32(
            x.data_ptr<float>(), weight.defined() ? weight.data_ptr<float>() : nullptr,
            bias.defined() ? bias.data_ptr<float>() : nullptr, out.data_ptr<float>(),
            _saved_mean.data_ptr<float>(), _saved_rstd.data_ptr<float>(), rows, cols, _eps);
#endif
    }
    return {out};
}

std::vector<Tensor> LayerNormOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& dY     = grad_outputs[0];
    const Tensor& X      = this->inputs()[0];
    const Tensor& weight = this->inputs()[1];  // 可能为空
    const Tensor& bias   = this->inputs()[2];  // 可能为空

    size_t cols = _normalized_shape.elements();
    size_t rows = X.element_num() / cols;

    // 1. 初始化梯度张量
    Tensor dX = Tensor::empty(X.shape(), X.options());
    Tensor dWeight;
    Tensor dBias;

    bool has_weight = weight.defined() && weight.element_num() > 0;
    if (has_weight) {
        // dWeight 和 dBias 需要被累加，所以必须用 zeros 初始化
        dWeight = Tensor::zeros(weight.shape(), weight.options());
        dBias   = Tensor::zeros(bias.shape(), bias.options());
    }

    if (X.device().isCpu()) {
        const float* dy_ptr = dY.data_ptr<float>();
        const float* x_ptr  = X.data_ptr<float>();
        const float* w_ptr  = has_weight ? weight.data_ptr<float>() : nullptr;

        float* dx_ptr = dX.data_ptr<float>();
        float* dw_ptr = has_weight ? dWeight.data_ptr<float>() : nullptr;
        float* db_ptr = has_weight ? dBias.data_ptr<float>() : nullptr;

        const float* mean_ptr = _saved_mean.data_ptr<float>();
        const float* rstd_ptr = _saved_rstd.data_ptr<float>();

        float f_cols = static_cast<float>(cols);

        // 按行（Batch * Seq_len）进行反向传播
        for (size_t r = 0; r < rows; ++r) {
            const float* curr_dy = dy_ptr + r * cols;
            const float* curr_x  = x_ptr + r * cols;
            float* curr_dx       = dx_ptr + r * cols;

            float mean = mean_ptr[r];
            float rstd = rstd_ptr[r];

            float sum_dy_w      = 0.0f;
            float sum_dy_w_xhat = 0.0f;

            // 第一遍循环：计算辅助的 Sum
            for (size_t c = 0; c < cols; ++c) {
                float x_hat = (curr_x[c] - mean) * rstd;
                float dy_w  = curr_dy[c] * (w_ptr ? w_ptr[c] : 1.0f);

                sum_dy_w += dy_w;
                sum_dy_w_xhat += dy_w * x_hat;
            }

            // 第二遍循环：计算 dX，并累加 dWeight 和 dBias
            float factor = rstd / f_cols;
            for (size_t c = 0; c < cols; ++c) {
                float x_hat = (curr_x[c] - mean) * rstd;
                float dy_w  = curr_dy[c] * (w_ptr ? w_ptr[c] : 1.0f);

                // 终极公式：求导 dX
                curr_dx[c] = factor * (f_cols * dy_w - sum_dy_w - x_hat * sum_dy_w_xhat);

                // 累加放射变换参数的梯度 (跨越所有 Rows 进行累加)
                if (has_weight) {
                    dw_ptr[c] += curr_dy[c] * x_hat;
                    db_ptr[c] += curr_dy[c];
                }
            }
        }
    } else if (X.device().isCuda()) {
#ifdef USE_CUDA
        const float* w_ptr = has_weight ? weight.data_ptr<float>() : nullptr;
        float* dw_ptr      = has_weight ? dWeight.data_ptr<float>() : nullptr;
        float* db_ptr      = has_weight ? dBias.data_ptr<float>() : nullptr;

        cuda::launch_layernorm_backward_float32(dY.data_ptr<float>(), X.data_ptr<float>(), w_ptr,
                                                _saved_mean.data_ptr<float>(),
                                                _saved_rstd.data_ptr<float>(), dX.data_ptr<float>(),
                                                dw_ptr, db_ptr, rows, cols, has_weight);
#else
        MINIDL_THROW_RUNTIME("Compiled without CUDA support!");
#endif
    }

    // 必须极其严格地返回与 inputs() 数量和顺序完全一致的梯度！
    return {dX, dWeight, dBias};
}

Tensor LayerNormOp::apply(const Tensor& x, const Tensor& weight, const Tensor& bias,
                          const Shape& normalized_shape, float eps) {
    auto op = std::make_shared<LayerNormOp>(normalized_shape, eps);
    return (*op)({x, weight, bias})[0];
}

}  // namespace miniDL