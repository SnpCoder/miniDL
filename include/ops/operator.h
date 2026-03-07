#ifndef __MINIDL_OPERATOR_H__
#define __MINIDL_OPERATOR_H__

#include <memory>
#include <vector>

#include "../core/tensor.h"
namespace miniDL {
class Operator : public std::enable_shared_from_this<Operator> {
   protected:
    std::vector<Tensor> _inputs;

    // weak_ptr to avoid circular reference
    // e.g. Operator -> output -> Tensor -> creator -> Operator
    std::vector<std::weak_ptr<TensorImpl>> _outputs;

   public:
    virtual ~Operator() = default;

    std::vector<Tensor> operator()(const std::vector<Tensor>& inputs) {
        _inputs                     = inputs;
        std::vector<Tensor> outputs = this->forward(inputs);

        bool requires_grad = false;
        for (const auto& in : inputs) {
            if (in.defined() && in.impl()->requires_grad()) {
                requires_grad = true;
                break;
            }
        }

        if (requires_grad) {
            _outputs.clear();
            for (auto& out : outputs) {
                out.impl()->set_requires_grad(true);
                out.impl()->set_creator(this->shared_from_this());
                _outputs.push_back(out.shared_impl());
            }
        }
        return outputs;
    }

    virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs)        = 0;
    virtual std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) = 0;

    const std::vector<Tensor>& inputs() const { return _inputs; }
    const std::vector<std::weak_ptr<TensorImpl>>& outputs() const { return _outputs; }
};
}  // namespace miniDL
#endif  // __MINIDL_OPERATOR_H__