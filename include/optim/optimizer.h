#ifndef __MINIDL_OPTIM_OPTIMIZER_H__
#define __MINIDL_OPTIM_OPTIMIZER_H__
#include <vector>

#include "../core/tensor.h"

namespace miniDL {
namespace optim {

class Optimizer {
   protected:
    // 工业级设计：直接持有网络参数的副本（里面是 shared_ptr，所以轻量且同步）
    std::vector<Tensor> _parameters;
    float _lr;  // 学习率

   public:
    Optimizer(const std::vector<Tensor>& parameters, float lr) : _parameters(parameters), _lr(lr) {}

    virtual ~Optimizer() = default;

    // 核心虚函数：每个具体的优化器自己去实现更新逻辑
    virtual void step() = 0;

    // PyTorch 经典接口：清空所有托管参数的梯度
    virtual void zero_grad() {
        for (auto& p : _parameters) {
            if (p.defined() && p.grad().defined()) {
                // 彻底销毁梯度图，释放显存
                p.impl()->set_grad(nullptr);
            }
        }
    }
};

}  // namespace optim
}  // namespace miniDL

#endif  // __MINIDL_OPTIM_OPTIMIZER_H__