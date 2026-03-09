#pragma once
#include "../operator.h"

namespace miniDL {

class GeluOp : public Operator {
   public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    static Tensor apply(const Tensor& x) {
        auto op = std::make_shared<GeluOp>();
        return (*op)({x})[0];
    }
};

}  // namespace miniDL