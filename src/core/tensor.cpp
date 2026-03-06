#include "../../include/core/tensor.h"

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

void Tensor::backward() {
    // TODO: implement autograd backward
}
}  // namespace miniDL