#pragma once
#include "../operator.h"

namespace miniDL {

class ReshapeOp : public Operator {
   private:
    Shape _new_shape;  // 前向传播需要的新形状
    Shape _old_shape;  // 反向传播需要的旧形状

   public:
    // 构造函数：把状态参数存起来
    ReshapeOp(const Shape& new_shape, const Shape& old_shape)
        : _new_shape(new_shape), _old_shape(old_shape) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    static Tensor apply(const Tensor& x, const Shape& new_shape);
};

}  // namespace miniDL