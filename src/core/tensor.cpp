#include "../../include/core/tensor.h"

#include <functional>
#include <unordered_set>

#include "../../include/ops/cuda/fill_cuda.cuh"
#include "../../include/ops/cuda/math/add_cuda.cuh"
#include "../../include/ops/math/add.h"
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

    // calculate total bytes to set 0
    void* ptr          = t.impl()->data();
    size_t total_bytes = t.element_num() * get_element_size(t.data_type());

    if (options.device().isCpu()) {
        std::memset(ptr, 0, total_bytes);
    } else if (options.device().isCuda()) {
#ifdef USE_CUDA
        cudaError_t err = cudaMemset(ptr, 0, total_bytes);
        if (err != cudaSuccess) {
            MINIDL_THROW_RUNTIME("cudaMemset failed: {}", cudaGetErrorString(err));
        }
#else
        MINIDL_THROW_RUNTIME("MiniDL is compiled without CUDA support");
#endif
    } else {
        MINIDL_THROW_INVALID_ARG("Unsupported device type: {}", options.device().to_string());
    }

    return t;
}

// one-filled tensor
Tensor Tensor::ones(const Shape& shape, const TensorOptions& options) {
    Tensor t = Tensor::empty(shape, options);

    // calculate total bytes to set 1
    float* ptr = t.data_ptr<float>();
    size_t n   = t.element_num();

    if (options.device().isCpu()) {
        std::fill(ptr, ptr + n, 1.0f);
    } else if (options.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_fill_kernel_float32(ptr, 1.0f, t.element_num());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            MINIDL_THROW_RUNTIME("cudaMemset failed: {}", cudaGetErrorString(err));
        }
#else
        MINIDL_THROW_RUNTIME("MiniDL is compiled without CUDA support");
#endif
    } else {
        MINIDL_THROW_INVALID_ARG("Unsupported device type: {}", options.device().to_string());
    }

    return t;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (this->shape() != other.shape()) {
        MINIDL_THROW_INVALID_ARG("Tensor operator+= requires matching shapes.");
    }
    if (this->device() != other.device()) {
        MINIDL_THROW_INVALID_ARG("Tensor operator+= requires same device.");
    }

    size_t n               = this->element_num();
    float* this_ptr        = this->data_ptr<float>();
    const float* other_ptr = other.data_ptr<float>();

    if (this->device().isCpu()) {
        for (size_t i = 0; i < n; ++i) { this_ptr[i] += other_ptr[i]; }
    } else if (this->device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_add_inplace_kernel_float32(this_ptr, other_ptr, n);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }

    return *this;
}

Tensor& Tensor::operator+=(float scalar) {
    size_t n        = this->element_num();
    float* this_ptr = this->data_ptr<float>();

    if (this->device().isCpu()) {
        for (size_t i = 0; i < n; ++i) this_ptr[i] += scalar;
    } else if (this->device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_add_scalar_inplace_kernel_float32(this_ptr, scalar, n);
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return *this;
}

Tensor Tensor::clone() const {
    if (!defined()) return Tensor();

    TensorOptions clone_opts = _impl->options();
    clone_opts.requiresGrad(false);  // avoid deadlock in backward propagation

    Tensor out         = Tensor::empty(this->shape(), clone_opts);
    size_t total_bytes = this->element_num() * get_element_size(this->data_type());

    if (this->device().isCpu()) {
        std::memcpy(out.impl()->data(), this->impl()->data(), total_bytes);
    } else if (this->device().isCuda()) {
#ifdef USE_CUDA
        cudaError_t err = cudaMemcpy(out.impl()->data(), this->impl()->data(), total_bytes,
                                     cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            MINIDL_THROW_RUNTIME("cudaMemcpy failed during clone: {}", cudaGetErrorString(err));
        }
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return out;
}

Tensor Tensor::to(Device dev) const {
    if (!defined()) { return Tensor(); }

    if (device() == dev) { return *this; }

    auto new_storage          = _impl->storage()->toDevice(dev);
    TensorOptions new_options = _impl->options();
    new_options.device(dev);

    auto new_impl =
        std::make_shared<TensorImpl>(std::move(new_storage), _impl->shape(), _impl->strides(),
                                     _impl->storage_offset(), new_options);

    return Tensor(std::move(new_impl));
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