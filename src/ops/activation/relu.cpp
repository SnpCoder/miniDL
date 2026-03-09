#include "../../../include/ops/activation/relu.h"

#ifdef USE_CUDA
#include "../../../include/ops/cuda/activation/relu_cuda.cuh"
#endif

namespace miniDL {

std::vector<Tensor> ReluOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& x = inputs[0];
    Tensor out      = Tensor::empty(x.shape(), x.options());

    if (x.device().isCpu()) {
        const float* x_ptr = x.data_ptr<float>();
        float* out_ptr     = out.data_ptr<float>();
        for (size_t i = 0; i < x.element_num(); ++i) {
            out_ptr[i] = x_ptr[i] > 0.0f ? x_ptr[i] : 0.0f;
        }
    } else if (x.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_relu_forward_float32(x.data_ptr<float>(), out.data_ptr<float>(),
                                          x.element_num());
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return {out};
}

std::vector<Tensor> ReluOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];
    const Tensor& x        = this->inputs()[0];  // 必须拿到前向的输入来判断 x > 0
    Tensor grad_x          = Tensor::empty(x.shape(), x.options());

    if (x.device().isCpu()) {
        const float* x_ptr  = x.data_ptr<float>();
        const float* go_ptr = grad_out.data_ptr<float>();
        float* gx_ptr       = grad_x.data_ptr<float>();
        for (size_t i = 0; i < x.element_num(); ++i) {
            gx_ptr[i] = x_ptr[i] > 0.0f ? go_ptr[i] : 0.0f;
        }
    } else if (x.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_relu_backward_float32(x.data_ptr<float>(), grad_out.data_ptr<float>(),
                                           grad_x.data_ptr<float>(), x.element_num());
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return {grad_x};
}

}  // namespace miniDL