#pragma once
#include "../ops/nn/embedding.h"
#include "module.h"

namespace miniDL {
namespace nn {

class Embedding : public Module {
   public:
    size_t _num_embeddings;
    size_t _embedding_dim;
    Tensor weight;

    // 构造函数
    Embedding(size_t num_embeddings, size_t embedding_dim, Device device = Device("cpu"))
        : _num_embeddings(num_embeddings), _embedding_dim(embedding_dim) {
        auto opt = miniDL::device(device).requiresGrad(true);

        // 按照正态分布初始化权重 (均值=0.0, 标准差=1.0)
        weight = Tensor::randn(Shape({num_embeddings, embedding_dim}), 0.0f, 1.0f, opt);

        // 注册到网络里，接收优化器的更新
        register_parameter("weight", weight);
    }

    Tensor forward(const Tensor& indices) { return EmbeddingOp::apply(indices, weight); }

    Tensor operator()(const Tensor& indices) { return forward(indices); }
};

}  // namespace nn
}  // namespace miniDL