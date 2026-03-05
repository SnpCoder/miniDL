#ifndef __MINIDL_SCALAR_H__
#define __MINIDL_SCALAR_H__

#include "basicTypes.h"
namespace miniDL {

class Scalar {
   private:
    union alignas(8) {
        float f32;
        double f64;
        int8_t i8;
        int32_t i32;
        int64_t i64;
        uint8_t u8;
    } v;

    DataType _dtype;

   public:
    Scalar() : _dtype(DataType::kFloat32) { v.f32 = 0.0f; }

    Scalar(float val) : _dtype(DataType::kFloat32) { v.f32 = val; }
    Scalar(double val) : _dtype(DataType::kFloat64) { v.f64 = val; }
    Scalar(int8_t val) : _dtype(DataType::kInt8) { v.i8 = val; }
    Scalar(int32_t val) : _dtype(DataType::kInt32) { v.i32 = val; }
    Scalar(int64_t val) : _dtype(DataType::kInt64) { v.i64 = val; }
    Scalar(uint8_t val) : _dtype(DataType::kUInt8) { v.u8 = val; }
};
};  // namespace miniDL

#endif  // __MINIDL_SCALAR_H__