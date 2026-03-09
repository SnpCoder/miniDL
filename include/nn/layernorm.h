#pragma once
#include "../ops/norm/layernorm.h"
#include "module.h"

namespace miniDL {
namespace nn {

class LayerNorm : public Module {
   public:
    Shape _normalized_shape;
    float _eps;

    // fix:直接作为成员变量，避免基类查找和右值引用问题
    Tensor weight;
    Tensor bias;

    LayerNorm(const Shape& normalized_shape, float eps = 1e-5, Device device = Device("cpu"))
        : _normalized_shape(normalized_shape), _eps(eps) {
        auto opt = miniDL::device(device).requiresGrad(true);

        weight = Tensor::ones(normalized_shape, opt);
        bias   = Tensor::zeros(normalized_shape, opt);

        register_parameter("weight", weight);
        register_parameter("bias", bias);
    }

    Tensor forward(const Tensor& x) {
        // 直接传入成员变量！
        return LayerNormOp::apply(x, weight, bias, _normalized_shape, _eps);
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

}  // namespace nn
}  // namespace miniDL