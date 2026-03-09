#pragma once
#include "../operator.h"

namespace miniDL {

class ReluOp : public Operator {
   public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    // 静态语法糖，方便 Module 调用并自动构建计算图
    static Tensor apply(const Tensor& x) {
        auto op = std::make_shared<ReluOp>();
        return (*op)({x})[0];
    }
};

}  // namespace miniDL