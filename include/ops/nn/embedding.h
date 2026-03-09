#pragma once
#include "../operator.h"

namespace miniDL {

class EmbeddingOp : public Operator {
   private:
    size_t _vocab_size;
    size_t _embed_dim;

   public:
    EmbeddingOp(size_t vocab_size, size_t embed_dim)
        : _vocab_size(vocab_size), _embed_dim(embed_dim) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    static Tensor apply(const Tensor& indices, const Tensor& weight);
};

}  // namespace miniDL