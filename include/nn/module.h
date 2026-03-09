#ifndef __MINIDL_NN_MODULE_H__
#define __MINIDL_NN_MODULE_H__
#include <memory>
#include <string>
#include <utility>  // for std::pair
#include <vector>

#include "../core/tensor.h"

namespace miniDL {
namespace nn {

class Module {
   protected:
    // 【PyTorch 工业级做法】：使用 Vector<Pair> 保证绝对的插入顺序！
    std::vector<std::pair<std::string, Tensor*>> _parameters;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> _modules;

    bool _is_training = true;

   public:
    virtual ~Module() = default;

    void register_parameter(const std::string& name, Tensor& tensor) {
        _parameters.push_back({name, &tensor});
    }

    void register_module(const std::string& name, std::shared_ptr<Module> module) {
        _modules.push_back({name, module});
    }

    virtual std::vector<Tensor> parameters() const {
        std::vector<Tensor> params;

        for (const auto& kv : _parameters) {
            if (kv.second && kv.second->defined() && kv.second->requires_grad()) {
                params.push_back(*(kv.second));
            }
        }

        for (const auto& kv : _modules) {
            if (kv.second) {
                auto child_params = kv.second->parameters();
                params.insert(params.end(), child_params.begin(), child_params.end());
            }
        }
        return params;
    }

    virtual void to(const Device& device) {
        for (auto& kv : _parameters) {
            if (kv.second && kv.second->defined()) { *(kv.second) = kv.second->to(device); }
        }
        for (auto& kv : _modules) {
            if (kv.second) kv.second->to(device);
        }
    }

    virtual void zero_grad() {
        for (auto& kv : _parameters) {
            if (kv.second && kv.second->defined() && kv.second->grad().defined()) {
                kv.second->impl()->set_grad(nullptr);
            }
        }
        for (auto& kv : _modules) {
            if (kv.second) kv.second->zero_grad();
        }
    }

    virtual void train(bool mode = true) {
        _is_training = mode;
        for (auto& kv : _modules) {
            if (kv.second) kv.second->train(mode);
        }
    }
    virtual void eval() { train(false); }
    bool is_training() const { return _is_training; }
};

}  // namespace nn
}  // namespace miniDL

#endif  // __MINIDL_NN_MODULE_H__