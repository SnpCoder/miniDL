#ifndef __MINIDL_TENSOR_H__
#define __MINIDL_TENSOR_H__

#include <memory>

#include "tensorImpl.h"
namespace miniDL {
class Tensor {
   private:
    std::shared_ptr<TensorImpl> _impl;

   public:
    Tensor() : _impl(nullptr) {}

    Tensor(std::shared_ptr<TensorImpl> impl) : _impl(std::move(impl)) {}

    static Tensor empty(const Shape& shape, const TensorOptions& options = TensorOptions());
    static Tensor zeros(const Shape& shape, const TensorOptions& options = TensorOptions());

    bool defined() const { return _impl != nullptr; }
    const Shape& shape() const { return _impl->shape(); }
    size_t element_num() const { return _impl->element_num(); }
    Device device() const { return _impl->device(); }
    DataType data_type() const { return _impl->data_type(); }

    TensorImpl* impl() const {
        return _impl.get();
    }  // unsafe, should be used by internal operators only

    template <typename T>
    T* data_ptr() const {
        return _impl->data_ptr<T>();
    }

    Tensor to(Device dev) const;
    void backward();
};
}  // namespace miniDL
#endif  // __MINIDL_TENSOR_H__