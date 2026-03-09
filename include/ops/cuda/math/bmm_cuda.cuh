#pragma once
#include <cstddef>

namespace miniDL {
namespace cuda {

void launch_bmm_auto_float32(const float* a, const float* b, float* out, size_t batch, size_t M,
                             size_t N, size_t K);

}  // namespace cuda
}  // namespace miniDL