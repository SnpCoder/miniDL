#ifndef __MINIDL_BASICTYPES_H__
#define __MINIDL_BASICTYPES_H__

#include <stdint.h>

#include <string>

#include "utils/exception.h"
namespace miniDL {

enum class DataType : int8_t {
    kFloat32 = 0,
    kFloat64 = 1,
    kInt8    = 2,
    kInt32   = 3,
    kInt64   = 4,
    kUInt8   = 5
};

static size_t get_element_size(DataType dt) {
    switch (dt) {
        case DataType::kFloat32:
            return 4;
        case DataType::kFloat64:
            return 8;
        case DataType::kInt32:
            return 4;
        case DataType::kInt64:
            return 8;
        case DataType::kInt8:
            return 1;
        case DataType::kUInt8:
            return 1;
        default:
            MINIDL_THROW_INVALID_ARG("Unsupported data type:{}", static_cast<int>(dt));
    }
};

// refer C++ origin type to DataType in compile-time
template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float> {
    static constexpr DataType type = DataType::kFloat32;
};

template <>
struct DataTypeTraits<double> {
    static constexpr DataType type = DataType::kFloat64;
};

template <>
struct DataTypeTraits<int8_t> {
    static constexpr DataType type = DataType::kInt8;
};

template <>
struct DataTypeTraits<int32_t> {
    static constexpr DataType type = DataType::kInt32;
};

template <>
struct DataTypeTraits<int64_t> {
    static constexpr DataType type = DataType::kInt64;
};

template <>
struct DataTypeTraits<uint8_t> {
    static constexpr DataType type = DataType::kUInt8;
};

enum class DeviceType { kCPU = 0, kGPU = 1 };

class Device {
   private:
    DeviceType _dtype;
    int _index;

   public:
    Device() : _dtype(DeviceType::kCPU), _index(0) {}
    Device(DeviceType dtype, int deviceId) : _dtype(dtype), _index(deviceId) {}

    explicit Device(const std::string& str) {
        if (str == "cpu") {
            _dtype = DeviceType::kCPU;
            _index = 0;
        } else if (str.find("cuda") == 0) {
            _dtype = DeviceType::kGPU;
            _index = 0;

            auto colonPos = str.find(':');
            if (colonPos != std::string::npos) { _index = std::stoi(str.substr(colonPos + 1)); }
        } else {
            MINIDL_THROW_INVALID_ARG("Invalid device name: {}", str);
        }
    }
    DeviceType type() const { return _dtype; }
    int index() const { return _index; }

    bool isCpu() const { return _dtype == DeviceType::kCPU; }
    bool isCuda() const { return _dtype == DeviceType::kGPU; }

    bool operator==(const Device& other) const {
        return _dtype == other._dtype && _index == other._index;
    }

    bool operator!=(const Device& other) const { return !(*this == other); }

    std::string to_string() {
        if (_dtype == DeviceType::kCPU) { return "cpu"; }
        return "cuda:" + std::to_string(_index);
    }
};
}  // namespace miniDL

#endif  // __MINIDL_BASICTYPES_H__