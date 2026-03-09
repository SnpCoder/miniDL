#include "../../../include/ops/nn/causal_mask.h"
#ifdef USE_CUDA
#include "../../../include/ops/cuda/nn/causal_mask_cuda.cuh"
#endif

namespace miniDL {
std::vector<Tensor> CausalMaskOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& x = inputs[0];
    Tensor out      = Tensor::empty(x.shape(), x.options());

    size_t seq_len     = x.shape()[x.ndim() - 1];  // 最后一个维度是 seq_len
    size_t batch_heads = x.element_num() / (seq_len * seq_len);

    if (x.device().isCpu()) {
        const float* in_ptr = x.data_ptr<float>();
        float* out_ptr      = out.data_ptr<float>();
        for (size_t b = 0; b < batch_heads; ++b) {
            for (size_t r = 0; r < seq_len; ++r) {
                for (size_t c = 0; c < seq_len; ++c) {
                    size_t idx   = b * seq_len * seq_len + r * seq_len + c;
                    out_ptr[idx] = (c > r) ? -1e9f : in_ptr[idx];
                }
            }
        }
    } else {
#ifdef USE_CUDA
        cuda::launch_causal_mask_forward_float32(x.data_ptr<float>(), out.data_ptr<float>(),
                                                 batch_heads, seq_len);
#endif
    }
    return {out};
}

std::vector<Tensor> CausalMaskOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];
    Tensor grad_in         = Tensor::empty(grad_out.shape(), grad_out.options());

    size_t seq_len     = grad_out.shape()[grad_out.ndim() - 1];
    size_t batch_heads = grad_out.element_num() / (seq_len * seq_len);

    if (grad_out.device().isCpu()) {
        const float* go_ptr = grad_out.data_ptr<float>();
        float* gi_ptr       = grad_in.data_ptr<float>();
        for (size_t b = 0; b < batch_heads; ++b) {
            for (size_t r = 0; r < seq_len; ++r) {
                for (size_t c = 0; c < seq_len; ++c) {
                    size_t idx  = b * seq_len * seq_len + r * seq_len + c;
                    gi_ptr[idx] = (c > r) ? 0.0f : go_ptr[idx];
                }
            }
        }
    } else {
#ifdef USE_CUDA
        cuda::launch_causal_mask_backward_float32(grad_out.data_ptr<float>(),
                                                  grad_in.data_ptr<float>(), batch_heads, seq_len);
#endif
    }
    return {grad_in};
}

Tensor CausalMaskOp::apply(const Tensor& x) {
    auto op = std::make_shared<CausalMaskOp>();
    return (*op)({x})[0];
}
}  // namespace miniDL