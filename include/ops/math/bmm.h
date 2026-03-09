#pragma once
#include "../operator.h"

namespace miniDL {

class BmmOp : public Operator {
   public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    static Tensor apply(const Tensor& a, const Tensor& b);
};

// 全局语法糖接口
Tensor bmm(const Tensor& a, const Tensor& b);

}  // namespace miniDL