#include "../../../include/ops/shape/transpose.h"

#include "../../../include/utils/exception.h"

#ifdef USE_CUDA
#include "../../../include/ops/cuda/shape/transpose_cuda.cuh"
#endif

namespace miniDL {

std::vector<Tensor> TransposeOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& a = inputs[0];

    if (a.ndim() != 2) {
        MINIDL_THROW_INVALID_ARG("miniDL TransposeOp currently only supports 2D matrices.");
    }

    int rows = a.shape()[0];
    int cols = a.shape()[1];

    // 内部申请物理内存，强制关闭这块新内存的梯度追踪（因为图追踪由 Operator 基类负责）
    TensorOptions trans_opts = a.options();
    trans_opts.requiresGrad(false);
    Tensor out =
        Tensor::empty(Shape({static_cast<size_t>(cols), static_cast<size_t>(rows)}), trans_opts);

    if (a.device().isCpu()) {
        const float* src = a.data_ptr<float>();
        float* dst       = out.data_ptr<float>();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) { dst[j * rows + i] = src[i * cols + j]; }
        }
    } else if (a.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_transpose_kernel_float32(a.data_ptr<float>(), out.data_ptr<float>(), rows,
                                              cols);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return {out};
}

std::vector<Tensor> TransposeOp::backward(const std::vector<Tensor>& grad_outputs) {
    // 【PyTorch 终极魔法】：转置的梯度，就是对梯度做转置！
    // 直接递归调用 apply，将其再次加入反向传播的计算图中！
    return {TransposeOp::apply(grad_outputs[0])};
}

Tensor TransposeOp::apply(const Tensor& a) {
    auto op = std::make_shared<TransposeOp>();
    return (*op)({a})[0];  // 触发 Operator 基类的 __call__，构建计算图
}

}  // namespace miniDL