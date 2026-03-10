#include "../../../include/ops/math/mul.h"

#include "../../../include/utils/exception.h"

#ifdef USE_CUDA
#include "../../../include/ops/cuda/math/mul_cuda.cuh"
#endif

namespace miniDL {
std::vector<Tensor> MulOp::forward(const std::vector<Tensor>& inputs) {
    if (inputs.size() != 2) { MINIDL_THROW_INVALID_ARG("MulOp requires exactly 2 input tensors"); }
    const Tensor& a = inputs[0];
    const Tensor& b = inputs[1];

    if (a.shape() != b.shape()) {
        MINIDL_THROW_INVALID_ARG("MulOp: input tensors must have the same shape");
    }
    if (a.device() != b.device()) {
        MINIDL_THROW_INVALID_ARG("MulOp: input tensors must be on the same device");
    }

    Tensor out = Tensor::empty(a.shape(), a.options());
    size_t n   = a.element_num();

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr     = out.data_ptr<float>();

    if (a.device().isCpu()) {
        for (size_t i = 0; i < n; ++i) { out_ptr[i] = a_ptr[i] * b_ptr[i]; }
    } else if (a.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_mul_kernel_float32(a_ptr, b_ptr, out_ptr, n);
#endif
    } else {
        MINIDL_THROW_INVALID_ARG("MulOp: unsupported device type {}", a.device().to_string());
    }
    return {out};
}

// grad_outputs:反向传播的输入
std::vector<Tensor> MulOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];
    // d(A * B)/dA = grad_out * B, d(A * B)/dB = grad_out * A
    Tensor grad_a = grad_out * this->inputs()[1];
    Tensor grad_b = grad_out * this->inputs()[0];
    return {grad_a, grad_b};
}

Tensor MulOp::apply(const Tensor& a, const Tensor& b) {
    auto op = std::make_shared<MulOp>();
    return (*op)({a, b})[0];
}

void MulOp::apply_inplace(Tensor& self, const Tensor& other) {
    if (self.shape() != other.shape()) {
        MINIDL_THROW_INVALID_ARG("MulOp inplace: shapes must match.");
    }
    if (self.device() != other.device()) {
        MINIDL_THROW_INVALID_ARG("MulOp inplace: tensors must be on the same device.");
    }

    size_t n           = self.element_num();
    float* s_ptr       = self.data_ptr<float>();
    const float* o_ptr = other.data_ptr<float>();

    if (self.device().isCpu()) {
        for (size_t i = 0; i < n; ++i) { s_ptr[i] *= o_ptr[i]; }
    } else if (self.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_mul_inplace_kernel_float32(s_ptr, o_ptr, n);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
}

std::vector<Tensor> MulScalarOp::forward(const std::vector<Tensor>& inputs) {
    if (inputs.size() != 1) MINIDL_THROW_INVALID_ARG("MulScalarOp requires 1 input.");
    const Tensor& a = inputs[0];

    Tensor out         = Tensor::empty(a.shape(), a.options());
    size_t n           = a.element_num();
    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr     = out.data_ptr<float>();

    if (a.device().isCpu()) {
        for (size_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] * _scalar;
    } else if (a.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_mul_scalar_kernel_float32(a_ptr, _scalar, out_ptr, n);
#endif
    }
    return {out};
}

std::vector<Tensor> MulScalarOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];
    // 微积分核心：d(A*c)/dA = grad_out * c
    return {grad_out * _scalar};
}

Tensor MulScalarOp::apply(const Tensor& a, float b) {
    auto op = std::make_shared<MulScalarOp>(b);
    return (*op)({a})[0];
}

void MulScalarOp::apply_inplace(Tensor& self, float scalar) {
    size_t n     = self.element_num();
    float* s_ptr = self.data_ptr<float>();

    if (self.device().isCpu()) {
        for (size_t i = 0; i < n; ++i) { s_ptr[i] *= scalar; }
    } else if (self.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_mul_scalar_inplace_kernel_float32(s_ptr, scalar, n);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    return MulOp::apply(a, b);
}
Tensor operator*(const Tensor& a, float b) {
    return MulScalarOp::apply(a, b);
}
Tensor operator*(float a, const Tensor& b) {
    return MulScalarOp::apply(b, a);
}
}  // namespace miniDL