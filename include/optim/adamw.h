#pragma once
#include <cmath>
#include <unordered_map>
#include <vector>

#include "../core/tensor.h"
#include "optimizer.h"

namespace miniDL {
namespace optim {

class AdamW : public Optimizer {
   private:
    float _beta1;
    float _beta2;
    float _eps;
    float _weight_decay;
    int _step_count;

    // 为每个权重张量维护状态 (使用 TensorImpl 的指针作为 Key)
    std::unordered_map<void*, Tensor> _m;
    std::unordered_map<void*, Tensor> _v;

   public:
    AdamW(const std::vector<Tensor>& parameters, float lr = 1e-3f, float beta1 = 0.9f,
          float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.01f)
        : Optimizer(parameters, lr)
        , _beta1(beta1)
        , _beta2(beta2)
        , _eps(eps)
        , _weight_decay(weight_decay)
        , _step_count(0) {}

    void zero_grad() override;
    void step() override;
};

}  // namespace optim
}  // namespace miniDL