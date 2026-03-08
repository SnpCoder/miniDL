#include "../../../include/ops/math/matmul.h"

#include "../../../include/utils/exception.h"

#ifdef USE_CUDA
#include "../../../include/ops/cuda/math/matmul_cuda.cuh"
#endif

namespace miniDL {

std::vector<Tensor> MatMulOp::forward(const std::vector<Tensor>& inputs) {
    if (inputs.size() != 2) {
        MINIDL_THROW_INVALID_ARG("MatMulOp requires exactly 2 input tensors");
    }
    const Tensor& a = inputs[0];
    const Tensor& b = inputs[1];

    if (a.ndim() != 2 || b.ndim() != 2) {
        MINIDL_THROW_INVALID_ARG("MatMulOp: requires 2D input tensors");
    }

    // M, K × K, N = M, N
    if (a.shape()[1] != b.shape()[0]) {
        MINIDL_THROW_INVALID_ARG("MatMulOp: A's columns must match B's rows for multiplication");
    }

    if (a.device() != b.device()) {
        MINIDL_THROW_INVALID_ARG("MatMulOp: input tensors must be on the same device");
    }

    int M = a.shape()[0];
    int K = a.shape()[1];  // = b.shape()[0]
    int N = b.shape()[1];

    Tensor out =
        Tensor::empty(Shape({static_cast<size_t>(M), static_cast<size_t>(N)}), a.options());

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr     = out.data_ptr<float>();

    if (a.device().isCpu()) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) { sum += a_ptr[i * K + k] * b_ptr[k * N + j]; }
                out_ptr[i * N + j] = sum;
            }
        }
    } else if (a.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_matmul_auto_float32(a_ptr, b_ptr, out_ptr, M, N, K);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return {out};
}

std::vector<Tensor> MatMulOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];
    const Tensor& A        = this->inputs()[0];
    const Tensor& B        = this->inputs()[1];

    // 注意：转置操作在物理底层往往使张量变得“不连续(non-contiguous)”
    // 因此这里调用的 transpose() 必须是我们上一轮写的那个“物理深拷贝版”
    Tensor grad_a = mm(grad_out, B.transpose());
    Tensor grad_b = mm(A.transpose(), grad_out);

    return {grad_a, grad_b};
}

Tensor MatMulOp::apply(const Tensor& a, const Tensor& b) {
    auto op = std::make_shared<MatMulOp>();
    return (*op)({a, b})[0];
}

Tensor mm(const Tensor& a, const Tensor& b) {
    return MatMulOp::apply(a, b);
}
}  // namespace miniDL