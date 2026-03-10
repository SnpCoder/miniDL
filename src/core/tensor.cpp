#include "../../include/core/tensor.h"

#include <functional>
#include <random>
#include <unordered_set>

#include "../../include/ops/cuda/fill_cuda.cuh"
#include "../../include/ops/cuda/math/add_cuda.cuh"
#include "../../include/ops/cuda/math/mul_cuda.cuh"
#include "../../include/ops/factory.h"
#include "../../include/ops/math/add.h"
#include "../../include/ops/math/bmm.h"
#include "../../include/ops/math/mul.h"
#include "../../include/ops/memory/contiguous.h"
#include "../../include/ops/shape/permute.h"
#include "../../include/ops/shape/reshape.h"
#include "../../include/ops/shape/transpose.h"
#include "../../include/utils/log.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace miniDL {

// uninitialized tensor
Tensor Tensor::empty(const Shape& shape, const TensorOptions& options) {
    return Tensor(std::make_shared<TensorImpl>(shape, options));
}

// zero-filled tensor
Tensor Tensor::zeros(const Shape& shape, const TensorOptions& options) {
    Tensor t = Tensor::empty(shape, options);
    ops::fill_zeros(t);
    return t;
}

// one-filled tensor
Tensor Tensor::ones(const Shape& shape, const TensorOptions& options) {
    Tensor t = Tensor::empty(shape, options);
    ops::fill_ones(t);
    return t;
}

Tensor Tensor::uniform(const Shape& shape, float low, float high, const TensorOptions& options) {
    Tensor t = Tensor::empty(shape, options);
    ops::fill_uniform(t, low, high);
    return t;
}

Tensor Tensor::randn(const Shape& shape, float mean, float std, const TensorOptions& options) {
    Tensor t = Tensor::empty(shape, options);
    ops::fill_randn(t, mean, std);
    return t;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    AddOp::apply_inplace(*this, other);
    return *this;
}

Tensor& Tensor::operator+=(float scalar) {
    AddScalarOp::apply_inplace(*this, scalar);
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    MulOp::apply_inplace(*this, other);
    return *this;
}

Tensor& Tensor::operator*=(float scalar) {
    MulScalarOp::apply_inplace(*this, scalar);
    return *this;
}

Tensor Tensor::clone() const {
    if (!defined()) return Tensor();
    return Tensor(_impl->clone());
}

Tensor Tensor::to(Device dev) const {
    if (!defined()) { return Tensor(); }

    if (device() == dev) { return *this; }

    return Tensor(_impl->to(dev));
}

// auto grad
void Tensor::backward() {
    if (!requires_grad()) {
        MINIDL_THROW_RUNTIME("Cannot call backward() on a tensor that does not require grad");
    }

    // initialize the begining gradient: dL / dL = 1
    // if grad isn't defined, we give it a all-one tensor
    if (!this->grad().defined()) {
        Tensor _ones = Tensor::ones(
            this->shape(),
            miniDL::device(this->device()).requiresGrad(false));  // set requiresGrad(false)!
        _impl->set_grad(_ones.shared_impl());
    }

    // topo-sort
    std::vector<std::shared_ptr<Operator>> topo_order;
    std::unordered_set<Operator*> visited;

    std::function<void(std::shared_ptr<Operator>)> build_topo = [&](std::shared_ptr<Operator> op) {
        if (!op || visited.count(op.get())) return;
        visited.insert(op.get());

        for (const auto& input : op->inputs()) {
            if (input.defined() && input.impl()->creator()) { build_topo(input.impl()->creator()); }
        }

        // visit all father node, and add itself
        topo_order.push_back(op);
    };

    build_topo(_impl->creator());
    MINIDL_INFO("Autograd Engine started. Total operators in graph: {}", topo_order.size());

    // 从队列尾部 (Loss 节点附近) 往头部 (网络输入层) 倒序遍历
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        auto op = *it;

        // a. 收集该算子所有输出张量的梯度
        std::vector<Tensor> grad_outputs;
        for (const auto& weak_out : op->outputs()) {
            if (auto out_impl = weak_out.lock()) {
                Tensor out_tensor(out_impl);
                if (out_tensor.grad().defined()) {
                    grad_outputs.push_back(out_tensor.grad());
                } else {
                    MINIDL_WARN("A required gradient is missing during backward propagation.");
                }
            }
        }

        // b. 调用算子的 backward 接口计算对输入的偏导数
        // 比如 Add 算子会把梯度 1:1 地复制两份退回来
        std::vector<Tensor> grad_inputs = op->backward(grad_outputs);

        for (size_t i = 0; i < op->inputs().size(); ++i) {
            const Tensor& input = op->inputs()[i];

            if (input.requires_grad()) {
                if (!input.grad().defined()) {
                    // 第一次分发到梯度，直接赋值
                    input.impl()->set_grad(grad_inputs[i].clone().shared_impl());
                } else {
                    // 如果这个张量被用在多处 (比如 C = A + A)，梯度需要累加！
                    // 我们直接使用重载的 operator+= 来完成这个数学操作
                    input.grad() += grad_inputs[i];
                }
            }
        }
    }
}

Tensor Tensor::transpose() const {
    return TransposeOp::apply(*this);
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    return ReshapeOp::apply(*this, new_shape);
}

Tensor Tensor::permute(const std::vector<size_t>& dims) const {
    return PermuteOp::apply(*this, dims);
}

Tensor Tensor::bmm(const Tensor& other) const {
    return BmmOp::apply(*this, other);
}

Tensor Tensor::contiguous() const {
    return ContiguousOp::apply(*this);
}

std::string Tensor::to_string() const {
    if (!defined()) return "Tensor(Undefined)";

    // if tensor is in cuda, we need to move it to cpu first
    if (device().isCuda()) { return this->to(Device("cpu")).impl()->to_string(); }

    return _impl->to_string();
}

void Tensor::print() const {
    if (!defined()) {
        MINIDL_PRINT("Tensor(Undefined)");
        return;
    }
    MINIDL_PRINT("{}", to_string());
}
}  // namespace miniDL