#include "../../include/core/tensorImpl.h"

#include <iomanip>
#include <sstream>

#include "../../include/utils/exception.h"
#include "../../include/utils/log.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace miniDL {
void TensorImpl::compute_contiguous_strides() {
    size_t ndim = _shape.size();
    _strides.resize(ndim);
    if (ndim == 0) { return; }  // scalar

    size_t current_stride = 1;
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        _strides[i] = current_stride;
        current_stride *= _shape[i];
    }
}

// owner of storage
TensorImpl::TensorImpl(const Shape& shape, const TensorOptions& options)
    : _shape(shape)
    , _storage_offset(0)
    , _options(options)
    , _require_grad(options.require_grad())
    , _grad(nullptr)
    , _creator(nullptr) {
    compute_contiguous_strides();
    size_t total_bytes = _shape.elements() * get_element_size(_options.data_type());
    _storage           = Storage::create(total_bytes, _options.device());
}

// view of storage
TensorImpl::TensorImpl(std::shared_ptr<Storage> storage, const Shape& shape,
                       const std::vector<size_t>& strides, size_t offset,
                       const TensorOptions& options)
    : _storage(std::move(storage))
    , _shape(shape)
    , _strides(strides)
    , _storage_offset(offset)
    , _options(options)
    , _require_grad(options.require_grad())
    , _grad(nullptr)
    , _creator(nullptr) {}

void* TensorImpl::data() const {
    if (!_storage || !_storage->data()) { return nullptr; }
    return static_cast<char*>(_storage->data()) +
           _storage_offset * get_element_size(_options.data_type());
}

bool TensorImpl::is_contiguous() const {
    size_t ndim = _shape.size();
    if (ndim == 0) { return true; }  // scalar

    size_t expected_stride = 1;
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        if (_shape[i] == 1) { continue; }  // skip stride check for dimensions of size 1
        if (_strides[i] != expected_stride) { return false; }
        expected_stride *= _shape[i];
    }
    return true;
}

size_t TensorImpl::compute_local_offset(const std::vector<size_t>& indices) const {
    if (indices.size() != _shape.size()) {
        MINIDL_THROW_INVALID_ARG("Indices size {} does not match tensor dimension {}",
                                 indices.size(), _shape.size());
    }

    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= _shape[i]) {
            MINIDL_THROW_OUT_OF_RANGE("Index {} out of bounds for dimension {} with size {}",
                                      indices[i], i, _shape[i]);
        }
        offset += indices[i] * _strides[i];
    }
    return offset;
}

namespace {
template <typename T>
void format_tensor_recursive(std::ostringstream& oss, const TensorImpl* impl,
                             std::vector<size_t>& current_indices, size_t dim) {
    if (dim == impl->shape().size()) {
        // Reached a single element, print it
        oss << std::fixed << std::setprecision(4) << impl->item<T>(current_indices);
        return;
    }

    oss << "[";
    for (size_t i = 0; i < impl->shape()[dim]; ++i) {
        current_indices[dim] = i;
        format_tensor_recursive<T>(oss, impl, current_indices, dim + 1);
        if (i < impl->shape()[dim] - 1) { oss << ", "; }
    }
    oss << "]";
}
}  // anonymous namespace

std::shared_ptr<TensorImpl> TensorImpl::permute(const std::vector<size_t>& dims) const {
    if (dims.size() != ndim()) {
        MINIDL_THROW_INVALID_ARG("Permute dims count must match tensor ndim.");
    }

    std::vector<size_t> new_shape(ndim());
    std::vector<size_t> new_strides(ndim());

    // 【核心魔法】：新形状和新步长，仅仅是老形状和老步长的重新排列！
    // 底层物理内存地址 (_storage) 完全没变！
    for (size_t i = 0; i < ndim(); ++i) {
        new_shape[i]   = _shape[dims[i]];
        new_strides[i] = _strides[dims[i]];
    }

    // 返回一个新的 Impl，共享 storage！
    return std::make_shared<TensorImpl>(_storage, Shape(new_shape), new_strides, _storage_offset,
                                        _options);
}

std::shared_ptr<TensorImpl> TensorImpl::reshape(const Shape& new_shape) const {
    if (new_shape.elements() != element_num()) {
        MINIDL_THROW_INVALID_ARG("Reshape cannot change the total number of elements.");
    }

    // 严格的零拷贝 Reshape (类似于 PyTorch 的 view)，要求内存必须是连续的
    if (!is_contiguous()) {
        // 工业级框架在这里会触发 deep copy (contiguous())，
        // 但为了我们现阶段的高效与严谨，我们直接报错，逼迫上层保证连续性
        MINIDL_THROW_RUNTIME("Zero-copy reshape requires a contiguous tensor.");
    }

    // 因为数据是连续的，我们只需要根据新 shape 重新计算标准的递减 strides 即可
    std::vector<size_t> new_strides(new_shape.size());
    size_t stride = 1;
    for (int i = new_shape.size() - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= new_shape[i];
    }

    return std::make_shared<TensorImpl>(_storage, new_shape, new_strides, _storage_offset,
                                        _options);
}

std::shared_ptr<TensorImpl> TensorImpl::clone() const {
    TensorOptions clone_opts = _options;
    clone_opts.requiresGrad(false);  // 克隆产生的节点默认不带梯度

    auto new_impl      = std::make_shared<TensorImpl>(_shape, clone_opts);
    size_t total_bytes = element_num() * get_element_size(this->data_type());

    if (this->device().isCpu()) {
        std::memcpy(new_impl->data(), this->data(), total_bytes);
    } else if (_options.device().isCuda()) {
#ifdef USE_CUDA
        cudaError_t err =
            cudaMemcpy(new_impl->data(), this->data(), total_bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            MINIDL_THROW_RUNTIME("cudaMemcpy failed during clone: {}", cudaGetErrorString(err));
        }
#else
        MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
    }
    return new_impl;
}

std::shared_ptr<TensorImpl> TensorImpl::to(Device dev) const {
    // the same divice judgement has done by Tensor

    auto new_storage          = _storage->toDevice(dev);
    TensorOptions new_options = _options;
    new_options.device(dev);

    return std::make_shared<TensorImpl>(std::move(new_storage), _shape, _strides, _storage_offset,
                                        new_options);
}

// pytorch-like tensor string representation, e.g. [[1.0000, 2.0000], [3.0000, 4.0000]]
std::string TensorImpl::to_string() const {
    if (_options.device().isCuda()) {
        MINIDL_THROW_RUNTIME("Cannot directly format a TensorImpl on CUDA. Move it to CPU first.");
    }

    std::ostringstream oss;
    oss << "tensor(";

    if (_shape.size() == 0) {
        // scalar
        if (_options.data_type() == DataType::kFloat32) {
            oss << std::fixed << std::setprecision(4) << item<float>({});
        } else if (_options.data_type() == DataType::kInt32) {
            oss << item<int32_t>({});
        }
    } else {
        std::vector<size_t> indices(_shape.size(), 0);
        if (_options.data_type() == DataType::kFloat32) {
            format_tensor_recursive<float>(oss, this, indices, 0);
        } else if (_options.data_type() == DataType::kInt32) {
            format_tensor_recursive<int32_t>(oss, this, indices, 0);
        } else {
            oss << "<Unsupported data type for string formatting>";
        }
    }

    oss << ", device='" << _options.device().to_string() << "'";
    if (_require_grad) { oss << ", requires_grad=true"; }
    oss << ")";
    return oss.str();
}

void TensorImpl::print() const {
    MINIDL_PRINT("{}", to_string());
}
}  // namespace miniDL