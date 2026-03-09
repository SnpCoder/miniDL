#ifndef __MINIDL_OPERATOR_LOSS_MSE_H__
#define __MINIDL_OPERATOR_LOSS_MSE_H__
#include "../operator.h"

namespace miniDL {

class MseLossOp : public Operator {
   public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    static Tensor apply(const Tensor& pred, const Tensor& target);
};

Tensor mse_loss(const Tensor& pred, const Tensor& target);

}  // namespace miniDL
#endif  // __MINIDL_OPERATOR_LOSS_MSE_H__