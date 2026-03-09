#pragma once
#include "module.h"

namespace miniDL {
namespace nn {

class Linear : public Module {
   private:
    Tensor _weight;
    Tensor _bias;
    bool _use_bias;

   public:
    // 构造函数：in_features (输入维度), out_features (输出维度)
    Linear(size_t in_features, size_t out_features, bool bias = true,
           Device device = Device("cpu"));

    // 前向传播
    Tensor forward(const Tensor& x);

    // C++ 语法糖：允许直接调用 layer(x) 代替 layer.forward(x)
    Tensor operator()(const Tensor& x) { return forward(x); }

    // 注意：我们完全删除了 parameters() 的重写！
    // 因为只要我们在构造函数里调用了 register_parameter，基类就会自动收集它们！
};

}  // namespace nn
}  // namespace miniDL