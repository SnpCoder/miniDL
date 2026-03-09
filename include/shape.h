#ifndef __MINIDL_SHAPE_H__
#define __MINIDL_SHAPE_H__

#include <stdint.h>

#include <stdexcept>
#include <vector>
namespace miniDL {

class Shape {
   private:
    std::vector<size_t> _dims;

   public:
    Shape() : _dims({}) {}

    Shape(const std::vector<size_t>& vec) : _dims(vec) {}

    Shape(std::initializer_list<size_t> _list) : _dims(_list) {}

    // non-const
    size_t& operator[](size_t index) {
        if (index >= _dims.size()) { throw std::out_of_range("Shape index out of range"); }
        return _dims[index];
    }

    // const, read-only
    const size_t& operator[](size_t index) const {
        if (index >= _dims.size()) { throw std::out_of_range("Shape index out of range"); }
        return _dims[index];
    }

    bool operator==(const Shape& other) const { return _dims == other._dims; }

    bool operator!=(const Shape& other) const { return !(*this == other); }

    size_t elements() const {
        if (_dims.empty()) { return 1; }

        size_t res = 1;
        for (size_t dim : _dims) {
            if (dim != 0 && SIZE_MAX / dim < res) {
                throw std::overflow_error("Shape Element total size is larger than SIZE_MAX");
            }
            res *= dim;
        }
        return res;
    }

    size_t size() const { return _dims.size(); }
    size_t ndim() const { return _dims.size(); }
    bool isScalar() { return _dims.empty(); }
};

};  // namespace miniDL

#endif  // __MINIDL_SHAPE_H__