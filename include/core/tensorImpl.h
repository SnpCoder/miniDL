#ifndef __MINIDL_TENSORIMPL_H__
#define __MINIDL_TENSORIMPL_H__

#include <memory>
#include <vector>

#include "../memory/storage.h"
#include "../shape.h"
#include "tensorOptions.h"
namespace miniDL {
class Operator;
class TensorImpl {
   private:
    std::shared_ptr<Storage> _storage;
    Shape _shape;
    std::vector<size_t> _strides;
    size_t _storage_offset;
    TensorOptions _options;

    bool _require_grad;
    std::shared_ptr<TensorImpl> _grad;
    std::shared_ptr<Operator> _creator;  // the begin of backpropagation

    void compute_contiguous_strides();

   public:
    TensorImpl(const Shape& shape, const TensorOptions& options);

    TensorImpl(std::shared_ptr<Storage> storage, const Shape& shape,
               const std::vector<size_t>& strides, size_t offset, const TensorOptions& options);

    ~TensorImpl() = default;

    // can not copy
    TensorImpl(const TensorImpl&)            = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;

    const Shape& shape() const { return _shape; }
    size_t ndim() const { return _shape.ndim(); }
    const std::vector<size_t>& strides() const { return _strides; }
    size_t storage_offset() const { return _storage_offset; }
    size_t element_num() const { return _shape.elements(); }
    const TensorOptions& options() const { return _options; }
    Storage* storage() const { return _storage.get(); }
    Device device() const { return _options.device(); }
    DataType data_type() const { return _options.data_type(); }

    size_t compute_local_offset(const std::vector<size_t>& indices) const;

    void* data() const;

    template <typename T>
    T* data_ptr() const {
        return static_cast<T*>(data());
    }

    template <typename T>
    T item(const std::vector<size_t>& indices) const {
        size_t offset = compute_local_offset(indices);
        // data() has already added storage_offset, so we can only add local stride offset here
        return static_cast<T*>(data())[offset];
    }

    bool is_contiguous() const;

    // auto grad
    bool requires_grad() const { return _require_grad; }
    void set_requires_grad(bool req) { _require_grad = req; };

    std::shared_ptr<TensorImpl> grad() const { return _grad; }
    void set_grad(std::shared_ptr<TensorImpl> g) { _grad = g; }

    std::shared_ptr<Operator> creator() const { return _creator; }
    void set_creator(std::shared_ptr<Operator> c) { _creator = c; }

    std::shared_ptr<TensorImpl> permute(const std::vector<size_t>& dims) const;
    std::shared_ptr<TensorImpl> reshape(const Shape& new_shape) const;

    std::string to_string() const;
    void print() const;
};
}  // namespace miniDL

#endif  // __MINIDL_TENSORIMPL_H__