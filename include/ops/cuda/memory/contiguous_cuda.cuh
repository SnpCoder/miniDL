#pragma once
#include <cstddef>

namespace miniDL {
namespace cuda {

#define MAX_DIMS 8

// 用于跨主机和设备传输的张量元数据结构
struct TensorMeta {
    int shape[MAX_DIMS];
    int strides[MAX_DIMS];
    int ndim;
};

void launch_contiguous_float32(const float* in, float* out, const TensorMeta& meta,
                               size_t total_elements);

}  // namespace cuda
}  // namespace miniDL