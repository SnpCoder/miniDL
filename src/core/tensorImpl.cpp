#include "../../include/core/tensorImpl.h"

#include <iomanip>
#include <sstream>

#include "../../include/utils/exception.h"
#include "../../include/utils/log.h"

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