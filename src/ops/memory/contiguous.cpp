#include "../../../include/ops/memory/contiguous.h"

#include "../../../include/utils/exception.h"

#ifdef USE_CUDA
#include "../../../include/ops/cuda/memory/contiguous_cuda.cuh"
#endif

namespace miniDL {

std::vector<Tensor> ContiguousOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& x = inputs[0];

    // 如果已经连续，直接返回自己！(Zero-overhead)
    if (x.impl()->is_contiguous()) { return {x}; }

    // 申请一块全新的、保证连续的物理内存
    Tensor out  = Tensor::empty(x.shape(), x.options());
    size_t ndim = x.ndim();
    size_t N    = x.element_num();

    if (ndim > 8) {
        MINIDL_THROW_RUNTIME("ContiguousOp currently only supports up to 8 dimensions.");
    }

    if (x.device().isCpu()) {
        const float* in_ptr = x.data_ptr<float>();
        float* out_ptr      = out.data_ptr<float>();
        const auto& shape   = x.shape();
        const auto& strides = x.impl()->strides();

        for (size_t i = 0; i < N; ++i) {
            size_t offset = 0;
            size_t curr   = i;
            for (int d = ndim - 1; d >= 0; --d) {
                offset += (curr % shape[d]) * strides[d];
                curr /= shape[d];
            }
            out_ptr[i] = in_ptr[offset];
        }
    } else if (x.device().isCuda()) {
#ifdef USE_CUDA
        cuda::TensorMeta meta;
        meta.ndim = ndim;
        for (size_t i = 0; i < ndim; ++i) {
            meta.shape[i]   = x.shape()[i];
            meta.strides[i] = x.impl()->strides()[i];
        }
        cuda::launch_contiguous_float32(x.data_ptr<float>(), out.data_ptr<float>(), meta, N);
#endif
    }
    return {out};
}

std::vector<Tensor> ContiguousOp::backward(const std::vector<Tensor>& grad_outputs) {
    // 【高能数学推导】：
    // 数学上，Y = X，只是内存排列变了。所以对矩阵而言，dY/dX 就是一个单位矩阵！
    // 我们只需要把上一层传回来的梯度原封不动传给 X 即可。
    return {grad_outputs[0]};
}

Tensor ContiguousOp::apply(const Tensor& x) {
    if (x.impl()->is_contiguous()) return x;  // 拦截器：如果连续就不进计算图了

    auto op = std::make_shared<ContiguousOp>();
    return (*op)({x})[0];
}

}  // namespace miniDL