#pragma once
#include "../core/tensor.h"

namespace miniDL {
namespace ops {

void fill_zeros(Tensor& t);
void fill_ones(Tensor& t);
void fill_uniform(Tensor& t, float low, float high);
void fill_randn(Tensor& t, float mean, float std);

}  // namespace ops
}  // namespace miniDL