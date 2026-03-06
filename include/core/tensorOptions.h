#ifndef __MINIDL_TENSOR_OPTIONS_H__
#define __MINIDL_TENSOR_OPTIONS_H__

#include "../basicTypes.h"

namespace miniDL {
class TensorOptions {
   private:
    Device _dev         = Device("cpu");
    DataType _data_type = DataType::kFloat32;
    bool _require_grad  = false;

   public:
    TensorOptions() = default;

    TensorOptions(DataType data_type) : _data_type(data_type) {}
    TensorOptions(Device dev) : _dev(dev) {}

    DataType data_type() const { return _data_type; }
    Device device() const { return _dev; }
    bool require_grad() const { return _require_grad; }

    bool operator==(const TensorOptions& other) const {
        return _data_type == other._data_type && _dev == other._dev &&
               _require_grad == other._require_grad;
    }

    bool operator!=(const TensorOptions& other) const { return !(*this == other); }

    TensorOptions& dataType(DataType dt) {
        _data_type = dt;
        return *this;
    }

    TensorOptions& device(Device dev) {
        _dev = dev;
        return *this;
    }

    TensorOptions& device(const std::string& str) {
        _dev = Device(str);
        return *this;
    }

    TensorOptions& requiresGrad(bool req) {
        _require_grad = req;
        return *this;
    }
};

// can called by chain: miniDL::device("cuda:0").requires_grad(true)
inline TensorOptions dataType(DataType dt) {
    return TensorOptions().dataType(dt);
}

inline TensorOptions device(Device dev) {
    return TensorOptions().device(dev);
}

inline TensorOptions device(const std::string& str) {
    return TensorOptions().device(str);
}

inline TensorOptions requireGrad(bool req) {
    return TensorOptions().requiresGrad(req);
}
}  // namespace miniDL
#endif  // __MINIDL_TENSOR_OPTIONS_H__