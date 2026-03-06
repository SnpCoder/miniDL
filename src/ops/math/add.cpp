#include "../../../include/ops/math/add.h"

#include "../../../include/utils/exception.h"
#include "../../../include/utils/log.h"

#ifdef USE_CUDA
#include "../../../include/ops/cuda/math/add_cuda.cuh"
#endif
namespace miniDL {
std::vector<Tensor> AddOp::forward(const std::vector<Tensor>& inputs) {
    if (inputs.size() != 2) { MINIDL_THROW_INVALID_ARG("AddOp requires exactly 2 inputs."); }
    const Tensor& a = inputs[0];
    const Tensor& b = inputs[1];

    if (a.shape() != b.shape()) {
        MINIDL_THROW_INVALID_ARG("AddOp: shapes must match. Broadcast not implemented yet.");
    }

    if (a.device() != b.device()) {
        MINIDL_THROW_INVALID_ARG("AddOp: tensors must be on the same device.");
    }

    Tensor out = Tensor::empty(a.shape(), a.options());
    size_t n   = a.element_num();

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr     = out.data_ptr<float>();

    if (a.device().isCpu()) {
        for (size_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] + b_ptr[i];
    } else if (a.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_add_kernel_float32(a_ptr, b_ptr, out_ptr, n);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return {out};
}

std::vector<Tensor> AddOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];
    // C = A + B, partial derivatives is 1
    return {grad_out, grad_out};
}

Tensor AddOp::apply(const Tensor& a, const Tensor& b) {
    auto op = std::make_shared<AddOp>();
    return (*op)({a, b})[0];
}

std::vector<Tensor> AddScalarOp::forward(const std::vector<Tensor>& inputs) {
    if (inputs.size() != 1) { MINIDL_THROW_INVALID_ARG("AddScalarOp requires only 1 input."); }
    const Tensor& a = inputs[0];

    Tensor out = Tensor::empty(a.shape(), a.options());
    size_t n   = a.element_num();

    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr     = out.data_ptr<float>();

    if (a.device().isCpu()) {
        for (size_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] + _scalar;
    } else if (a.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_add_scalar_kernel_float32(a_ptr, _scalar, out_ptr, n);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return {out};
}

std::vector<Tensor> AddScalarOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];
    // C = A + scalar，A's partial derivatives is 1，scalar is 0
    return {grad_out};
}

Tensor AddScalarOp::apply(const Tensor& a, float b) {
    auto op = std::make_shared<AddScalarOp>(b);
    return (*op)({a})[0];
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    return AddOp::apply(a, b);
}

Tensor operator+(const Tensor& a, float b) {
    return AddScalarOp::apply(a, b);
}

Tensor operator+(float a, const Tensor& b) {
    return AddScalarOp::apply(b, a);
}
}  // namespace miniDL