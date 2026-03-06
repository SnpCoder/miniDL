#ifndef __MINIDL_OPERATOR_ADD_H__
#define __MINIDL_OPERATOR_ADD_H__

#include "../operator.h"

namespace miniDL {
class AddOp : public Operator {
   public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    static Tensor apply(const Tensor& a, const Tensor& b);
};

class AddScalarOp : public Operator {
   private:
    float _scalar;

   public:
    explicit AddScalarOp(float scalar) : _scalar(scalar) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& input) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    static Tensor apply(const Tensor& a, float b);
};

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, float b);
Tensor operator+(float a, const Tensor& b);
}  // namespace miniDL

#endif  // __MINIDL_OPERATOR_ADD_H__
