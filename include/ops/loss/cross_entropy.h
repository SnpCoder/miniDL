#pragma once
#include "../operator.h"

namespace miniDL {

class CrossEntropyLossOp : public Operator {
   private:
    Tensor _softmax_probs;  // 缓存前向传播算出来的概率
    Tensor _target;         // 缓存真实标签

   public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    static Tensor apply(const Tensor& pred, const Tensor& target);
};

// 语法糖
Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target);

}  // namespace miniDL