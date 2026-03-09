#pragma once
#include "../operator.h"

namespace miniDL {

class LayerNormOp : public Operator {
   private:
    float _eps;
    Shape _normalized_shape;

    // 【架构精髓】：缓存前向传播计算出的均值和倒数标准差，反向传播求导极其依赖它们！
    Tensor _saved_mean;
    Tensor _saved_rstd;

   public:
    LayerNormOp(const Shape& normalized_shape, float eps = 1e-5)
        : _normalized_shape(normalized_shape), _eps(eps) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    static Tensor apply(const Tensor& x, const Tensor& weight, const Tensor& bias,
                        const Shape& normalized_shape, float eps = 1e-5);
};

}  // namespace miniDL