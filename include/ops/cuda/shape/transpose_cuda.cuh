#ifndef __MINIDL_OPERATOR_TRANSPOSE_CUDA_H__
#define __MINIDL_OPERATOR_TRANSPOSE_CUDA_H__
#include <cstddef>

namespace miniDL {
namespace cuda {

// 物理转置：将 [rows, cols] 的矩阵搬运为 [cols, rows]
void launch_transpose_kernel_float32(const float* in, float* out, int rows, int cols);

}  // namespace cuda
}  // namespace miniDL

#endif  // __MINIDL_OPERATOR_TRANSPOSE_CUDA_H__