#pragma once
#include "../operator.h"

namespace miniDL {

class SoftmaxOp : public Operator {
   private:
    // 【核心架构设计】：缓存前向传播计算出的概率分布 Y
    // 反向传播算雅可比矩阵的时候必须用到它！
    Tensor _saved_out;

   public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    static Tensor apply(const Tensor& x) {
        auto op = std::make_shared<SoftmaxOp>();
        return (*op)({x})[0];
    }
};

}  // namespace miniDL