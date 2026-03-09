#include "../../../include/ops/shape/reshape.h"

namespace miniDL {

std::vector<Tensor> ReshapeOp::forward(const std::vector<Tensor>& inputs) {
    // 在 forward 里执行真正的零拷贝逻辑！
    Tensor out = Tensor(inputs[0].impl()->reshape(_new_shape));
    return {out};
}

std::vector<Tensor> ReshapeOp::backward(const std::vector<Tensor>& grad_outputs) {
    // 反向传播：把传进来的梯度 Reshape 回原来的形状
    Tensor grad_in = Tensor(grad_outputs[0].impl()->reshape(_old_shape));
    return {grad_in};
}

Tensor ReshapeOp::apply(const Tensor& x, const Shape& new_shape) {
    // 实例化算子时传入状态参数
    auto op = std::make_shared<ReshapeOp>(new_shape, x.shape());
    // 完美复用基类 Operator 的魔法
    return (*op)({x})[0];
}

}  // namespace miniDL