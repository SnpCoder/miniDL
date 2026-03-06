#include "../../include/core/tensorImpl.h"

#include "../../include/utils/exception.h"

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
}  // namespace miniDL