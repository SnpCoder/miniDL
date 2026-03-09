#include "../../../include/ops/math/bmm.h"

#include "../../../include/ops/memory/contiguous.h"
#include "../../../include/utils/exception.h"

#ifdef USE_CUDA
#include "../../../include/ops/cuda/math/bmm_cuda.cuh"
#endif

namespace miniDL {

std::vector<Tensor> BmmOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& A = inputs[0];
    const Tensor& B = inputs[1];

    if (A.ndim() != 3 || B.ndim() != 3) {
        MINIDL_THROW_INVALID_ARG("BMM only supports 3D tensors [Batch, M, K] x [Batch, K, N].");
    }
    if (A.shape()[0] != B.shape()[0] || A.shape()[2] != B.shape()[1]) {
        MINIDL_THROW_INVALID_ARG("BMM shape mismatch!");
    }

    size_t batch = A.shape()[0];
    size_t M     = A.shape()[1];
    size_t K     = A.shape()[2];
    size_t N     = B.shape()[2];

    // 【极其重要的防线】：BMM 的指针运算假设内存是连续的！
    Tensor A_contig = A.impl()->is_contiguous() ? A : A.contiguous();
    Tensor B_contig = B.impl()->is_contiguous() ? B : B.contiguous();

    Tensor C = Tensor::empty(Shape({batch, M, N}), A.options());

    if (A.device().isCpu()) {
        const float* a_ptr = A_contig.data_ptr<float>();
        const float* b_ptr = B_contig.data_ptr<float>();
        float* c_ptr       = C.data_ptr<float>();

        for (size_t b = 0; b < batch; ++b) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        sum += a_ptr[b * M * K + m * K + k] * b_ptr[b * K * N + k * N + n];
                    }
                    c_ptr[b * M * N + m * N + n] = sum;
                }
            }
        }
    } else if (A.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_bmm_auto_float32(A_contig.data_ptr<float>(), B_contig.data_ptr<float>(),
                                      C.data_ptr<float>(), batch, M, N, K);
#endif
    }

    return {C};
}

std::vector<Tensor> BmmOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& dC = grad_outputs[0];
    const Tensor& A  = this->inputs()[0];
    const Tensor& B  = this->inputs()[1];

    // 反向传播数学公式:
    // dA = dC * B^T
    // dB = A^T * dC

    // 1. 获取不需要追踪梯度的、物理连续的转置矩阵！
    // 复用你写的底层魔法：先 permute({0, 2, 1}) 交换最后两个维度，再调用连续化！
    Tensor B_T       = Tensor(B.impl()->permute({0, 2, 1})).contiguous();
    Tensor A_T       = Tensor(A.impl()->permute({0, 2, 1})).contiguous();
    Tensor dC_contig = dC.impl()->is_contiguous() ? dC : dC.contiguous();

    // 2. 为了计算 dA 和 dB，我们直接复用前向的 C++ 逻辑 (不开计算图)
    auto bmm_compute = [](const Tensor& x, const Tensor& y) -> Tensor {
        BmmOp op;
        return op.forward({x, y})[0];
    };

    Tensor dA = bmm_compute(dC_contig, B_T);
    Tensor dB = bmm_compute(A_T, dC_contig);

    return {dA, dB};
}

Tensor BmmOp::apply(const Tensor& a, const Tensor& b) {
    auto op    = std::make_shared<BmmOp>();
    Tensor out = (*op)({a, b})[0];
    return out;
}

Tensor bmm(const Tensor& a, const Tensor& b) {
    return BmmOp::apply(a, b);
}

}  // namespace miniDL