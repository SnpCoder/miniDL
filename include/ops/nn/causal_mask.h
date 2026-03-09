#pragma once
#include "../operator.h"

namespace miniDL {
class CausalMaskOp : public Operator {
   public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    static Tensor apply(const Tensor& x);
};
}  // namespace miniDL