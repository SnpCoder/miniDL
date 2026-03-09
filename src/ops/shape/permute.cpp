#include "../../../include/ops/shape/permute.h"

namespace miniDL {
std::vector<Tensor> PermuteOp::forward(const std::vector<Tensor>& inputs) {
    // 执行零拷贝换位
    Tensor out = Tensor(inputs[0].impl()->permute(_dims));
    return {out};
}

std::vector<Tensor> PermuteOp::backward(const std::vector<Tensor>& grad_outputs) {
    const Tensor& grad_out = grad_outputs[0];

    // Permute 的反向传播：求出一个“逆向的排列顺序 (Inverse Permutation)”
    std::vector<size_t> inv_dims(_dims.size());
    for (size_t i = 0; i < _dims.size(); ++i) { inv_dims[_dims[i]] = i; }

    // 用逆向顺序把梯度再换位回来！
    Tensor grad_in = Tensor(grad_out.impl()->permute(inv_dims));
    return {grad_in};
}

Tensor PermuteOp::apply(const Tensor& x, const std::vector<size_t>& dims) {
    auto op = std::make_shared<PermuteOp>(dims);
    // 同样复用基类的标准分发逻辑
    return (*op)({x})[0];
}
}  // namespace miniDL