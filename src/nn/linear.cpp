#include "../../include/nn/linear.h"

#include <cmath>

#include "../../include/ops/math/add.h"
#include "../../include/ops/math/matmul.h"
#include "../../include/ops/shape/broadcast.h"  // 引入广播魔法

namespace miniDL {
namespace nn {

Linear::Linear(size_t in_features, size_t out_features, bool bias, Device device)
    : _use_bias(bias) {
    auto opt = miniDL::device(device).requiresGrad(true);

    float stdv = 1.0f / std::sqrt(static_cast<float>(in_features));

    // 【PyTorch 原生形状】：恢复为 [out_features, in_features]
    _weight = Tensor::uniform(Shape({out_features, in_features}), 0.0f, stdv, opt);
    this->register_parameter("weight", _weight);

    if (_use_bias) {
        _bias = Tensor::uniform(Shape({out_features}), 0.0f, stdv, opt);
        this->register_parameter("bias", _bias);
    }
}

Tensor Linear::forward(const Tensor& x) {
    // 【原生数学公式】：Y = X * W^T
    // 现在 transpose() 是一个算子了，计算图会通过 TransposeBackward 完美连通！
    Tensor out = mm(x, _weight.transpose());

    if (_use_bias) {
        Tensor broadcasted_bias = broadcast_to(_bias, out.shape());
        out                     = out + broadcasted_bias;
    }
    return out;
}

}  // namespace nn
}  // namespace miniDL