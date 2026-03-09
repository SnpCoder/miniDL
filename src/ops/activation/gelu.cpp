#include "../../../include/ops/activation/gelu.h"

#include <cmath>

#ifdef USE_CUDA
#include "../../../include/ops/cuda/activation/gelu_cuda.cuh"
#endif

namespace miniDL {

// 常量预计算
constexpr float SQRT_2_OVER_PI = 0.7978845608f;
constexpr float COEF           = 0.044715f;

std::vector<Tensor> GeluOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& x = inputs[0];
    Tensor out      = Tensor::empty(x.shape(), x.options());

    if (x.device().isCpu()) {
        const float* x_ptr = x.data_ptr<float>();
        float* out_ptr     = out.data_ptr<float>();
        for (size_t i = 0; i < x.element_num(); ++i) {
            float val   = x_ptr[i];
            float inner = SQRT_2_OVER_PI * (val + COEF * val * val * val);
            out_ptr[i]  = 0.5f * val * (1.0f + std::tanh(inner));
        }
    } else if (x.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_gelu_forward_float32(x.data_ptr<float>(), out.data_ptr<float>(),
                                          x.element_num());
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return {out};
}

std::vector<Tensor> GeluOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];
    const Tensor& x        = this->inputs()[0];
    Tensor grad_x          = Tensor::empty(x.shape(), x.options());

    if (x.device().isCpu()) {
        const float* x_ptr  = x.data_ptr<float>();
        const float* go_ptr = grad_out.data_ptr<float>();
        float* gx_ptr       = grad_x.data_ptr<float>();
        for (size_t i = 0; i < x.element_num(); ++i) {
            float val        = x_ptr[i];
            float inner      = SQRT_2_OVER_PI * (val + COEF * val * val * val);
            float tanh_inner = std::tanh(inner);

            // GELU 导数公式 (链式法则展开)
            float left  = 0.5f * (1.0f + tanh_inner);
            float sech2 = 1.0f - tanh_inner * tanh_inner;
            float right = 0.5f * val * sech2 * SQRT_2_OVER_PI * (1.0f + 3.0f * COEF * val * val);

            gx_ptr[i] = go_ptr[i] * (left + right);
        }
    } else if (x.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_gelu_backward_float32(x.data_ptr<float>(), grad_out.data_ptr<float>(),
                                           grad_x.data_ptr<float>(), x.element_num());
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return {grad_x};
}

}  // namespace miniDL