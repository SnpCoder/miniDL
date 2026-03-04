#ifndef __MINIDL_BASICTYPES_H__
#define __MINIDL_BASICTYPES_H__

#include <stdint.h>
namespace miniDL {

enum class DataType : int8_t {
    kFloat32 = 0,
    kFloat64 = 1,
    kInt8    = 2,
    kInt32   = 3,
    kInt64   = 4,
    kUInt8   = 5
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
}  // namespace miniDL

#endif  // __MINIDL_BASICTYPES_H__