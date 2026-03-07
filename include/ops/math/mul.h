#ifndef __MINIDL_OPERATOR_MUL_H__
#define __MINIDL_OPERATOR_MUL_H__
#include "../operator.h"

namespace miniDL {

class MulOp : public Operator {
   public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    static Tensor apply(const Tensor& a, const Tensor& b);
};

class MulScalarOp : public Operator {
   private:
    float _scalar;

   public:
    explicit MulScalarOp(float scalar) : _scalar(scalar) {}
    std::vector<Tensor> forward(const std::vector<Tensor>& input) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    static Tensor apply(const Tensor& a, float b);
};

// 运算符重载 (语法糖)
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, float b);
Tensor operator*(float a, const Tensor& b);

}  // namespace miniDL

#endif  // __MINIDL_OPERATOR_MUL_H__