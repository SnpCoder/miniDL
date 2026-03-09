#include "../../../include/ops/nn/embedding.h"

#include "../../../include/utils/exception.h"

#ifdef USE_CUDA
#include "../../../include/ops/cuda/nn/embedding_cuda.cuh"
#endif

namespace miniDL {

std::vector<Tensor> EmbeddingOp::forward(const std::vector<Tensor>& inputs) {
    const Tensor& indices = inputs[0];
    const Tensor& weight  = inputs[1];

    size_t total_tokens = indices.element_num();

    // 构建输出形状：在原形状最后加上 embed_dim
    std::vector<size_t> out_shape_vec = indices.shape().vec();
    out_shape_vec.push_back(_embed_dim);
    Shape out_shape(out_shape_vec);

    Tensor out = Tensor::empty(out_shape, weight.options());

    if (indices.device().isCpu()) {
        const float* idx_ptr = indices.data_ptr<float>();
        const float* w_ptr   = weight.data_ptr<float>();
        float* out_ptr       = out.data_ptr<float>();

        for (size_t i = 0; i < total_tokens; ++i) {
            int word_id = static_cast<int>(idx_ptr[i]);
            if (word_id >= 0 && word_id < _vocab_size) {
                for (size_t d = 0; d < _embed_dim; ++d) {
                    out_ptr[i * _embed_dim + d] = w_ptr[word_id * _embed_dim + d];
                }
            } else {
                for (size_t d = 0; d < _embed_dim; ++d) { out_ptr[i * _embed_dim + d] = 0.0f; }
            }
        }
    } else if (indices.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_embedding_forward_float32(indices.data_ptr<float>(), weight.data_ptr<float>(),
                                               out.data_ptr<float>(), total_tokens, _embed_dim,
                                               _vocab_size);
#endif
    }
    return {out};
}

std::vector<Tensor> EmbeddingOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];
    const Tensor& indices  = this->inputs()[0];
    const Tensor& weight   = this->inputs()[1];

    size_t total_tokens = indices.element_num();

    // 索引 (indices) 不需要计算梯度，只需给 weight 算梯度
    Tensor grad_indices;  // 留空
    // 【关键】：这里必须用 zeros 初始化，因为我们要累加梯度！
    Tensor grad_weight = Tensor::zeros(weight.shape(), weight.options());

    if (indices.device().isCpu()) {
        const float* go_ptr  = grad_out.data_ptr<float>();
        const float* idx_ptr = indices.data_ptr<float>();
        float* gw_ptr        = grad_weight.data_ptr<float>();

        for (size_t i = 0; i < total_tokens; ++i) {
            int word_id = static_cast<int>(idx_ptr[i]);
            if (word_id >= 0 && word_id < _vocab_size) {
                for (size_t d = 0; d < _embed_dim; ++d) {
                    // CPU 端直接累加
                    gw_ptr[word_id * _embed_dim + d] += go_ptr[i * _embed_dim + d];
                }
            }
        }
    } else if (indices.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_embedding_backward_float32(
            grad_out.data_ptr<float>(), indices.data_ptr<float>(), grad_weight.data_ptr<float>(),
            total_tokens, _embed_dim, _vocab_size);
#endif
    }
    return {grad_indices, grad_weight};
}

Tensor EmbeddingOp::apply(const Tensor& indices, const Tensor& weight) {
    auto op = std::make_shared<EmbeddingOp>(weight.shape()[0], weight.shape()[1]);
    return (*op)({indices, weight})[0];
}

}  // namespace miniDL