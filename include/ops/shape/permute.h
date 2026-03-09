#pragma once
#include "../operator.h"

namespace miniDL {
class PermuteOp : public Operator {
   private:
    std::vector<size_t> _dims;

   public:
    // 构造函数：把要换位的维度存起来
    PermuteOp(const std::vector<size_t>& dims) : _dims(dims) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    static Tensor apply(const Tensor& x, const std::vector<size_t>& dims);
};

}  // namespace miniDL