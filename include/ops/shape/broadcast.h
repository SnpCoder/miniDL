#ifndef __MINIDL_OPERATOR_BROADCASE_H__
#define __MINIDL_OPERATOR_BROADCASE_H__
#include "../operator.h"

namespace miniDL {

class BroadcastToOp : public Operator {
   private:
    Shape _target_shape;

   public:
    explicit BroadcastToOp(const Shape& target_shape) : _target_shape(target_shape) {}
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    static Tensor apply(const Tensor& x, const Shape& target_shape);
};

// 语法糖
Tensor broadcast_to(const Tensor& x, const Shape& target_shape);

}  // namespace miniDL

#endif