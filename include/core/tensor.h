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
    static Tensor ones(const Shape& shape, const TensorOptions& options = TensorOptions());
    static Tensor uniform(const Shape& shape, float low = 0.0f, float high = 1.0f,
                          const TensorOptions& options = TensorOptions());
    static Tensor randn(const Shape& shape, float mean = 0.0f, float std = 1.0f,
                        const TensorOptions& options = TensorOptions());

    bool defined() const { return _impl != nullptr; }
    const Shape& shape() const { return _impl->shape(); }
    size_t element_num() const { return _impl->element_num(); }
    size_t ndim() const { return _impl->shape().size(); }
    Device device() const { return _impl->device(); }
    DataType data_type() const { return _impl->data_type(); }
    const TensorOptions& options() const { return _impl->options(); }

    Tensor& operator+=(const Tensor& other);
    Tensor& operator+=(float scalar);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator*=(float scalar);

    TensorImpl* impl() const {
        return _impl.get();
    }  // unsafe, should be used by internal operators only

    std::shared_ptr<TensorImpl> shared_impl() const { return _impl; }
    Tensor clone() const;

    template <typename T>
    T* data_ptr() const {
        return _impl->data_ptr<T>();
    }

    template <typename T>
    T item(const std::vector<size_t>& indices) const {
        // if data is in cuda, we need to move it to cpu first before accessing the item
        if (device().isCuda()) { return this->to(Device("cpu")).impl()->item<T>(indices); }
        return _impl->item<T>(indices);
    }

    bool requires_grad() const { return _impl && _impl->requires_grad(); }
    Tensor grad() const {
        if (!_impl || !_impl->grad()) return Tensor();
        return Tensor(_impl->grad());
    }
    Tensor to(Device dev) const;
    void backward();
    Tensor transpose() const;
    Tensor reshape(const Shape& new_shape) const;
    Tensor permute(const std::vector<size_t>& dims) const;
    Tensor bmm(const Tensor& other) const;

    bool is_contiguous() const { return _impl->is_contiguous(); }
    Tensor contiguous() const;

    std::string to_string() const;
    void print() const;
};
}  // namespace miniDL
#endif  // __MINIDL_TENSOR_H__