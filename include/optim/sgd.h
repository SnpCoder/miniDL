#ifndef __MINIDL_OPTIM_SGD_H__
#define __MINIDL_OPTIM_SGD_H__
#include "optimizer.h"

namespace miniDL {
namespace optim {

class SGD : public Optimizer {
   private:
    float _momentum;
    float _weight_decay;

    // 状态缓冲区：为每个参数保存对应的动量 Tensor
    std::unordered_map<Tensor*, Tensor> _momentum_buffers;

   public:
    SGD(const std::vector<Tensor>& parameters, float lr, float momentum = 0.0f,
        float weight_decay = 0.0f)
        : Optimizer(parameters, lr), _momentum(momentum), _weight_decay(weight_decay) {}

    void step() override;
};

}  // namespace optim
}  // namespace miniDL

#endif