#pragma once
#include "../ops/activation/gelu.h"
#include "../ops/activation/relu.h"
#include "../ops/activation/softmax.h"
#include "module.h"

namespace miniDL {
namespace nn {

class ReLU : public Module {
   public:
    Tensor forward(const Tensor& x) { return ReluOp::apply(x); }
    Tensor operator()(const Tensor& x) { return forward(x); }
};

class GELU : public Module {
   public:
    Tensor forward(const Tensor& x) { return GeluOp::apply(x); }
    Tensor operator()(const Tensor& x) { return forward(x); }
};

class Softmax : public Module {
   public:
    Tensor forward(const Tensor& x) { return SoftmaxOp::apply(x); }
    Tensor operator()(const Tensor& x) { return forward(x); }
};

}  // namespace nn
}  // namespace miniDL