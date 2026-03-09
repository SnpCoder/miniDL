#ifndef __MINIDL_NN_LOSS_H__
#define __MINIDL_NN_LOSS_H__
#include "../ops/loss/mse.h"
#include "module.h"

namespace miniDL {
namespace nn {

class MSELoss : public Module {
   public:
    MSELoss() = default;

    Tensor forward(const Tensor& pred, const Tensor& target) { return mse_loss(pred, target); }

    Tensor operator()(const Tensor& pred, const Tensor& target) { return forward(pred, target); }
};

}  // namespace nn
}  // namespace miniDL
#endif  // __MINIDL_NN_LOSS_H__