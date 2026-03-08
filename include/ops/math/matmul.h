#ifndef __MINIDL_OPERATOR_MATMUL_H__
#define __MINIDL_OPERATOR_MATMUL_H__

#include "../operator.h"
namespace miniDL {
class MatMulOp : public Operator {
   public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    // C[M, N] = A[M, K] * B[K, N]
    static Tensor apply(const Tensor& a, const Tensor& b);
};

// 运算符重载
Tensor mm(const Tensor& a, const Tensor& b);
}  // namespace miniDL
#endif  // __MINIDL_OPERATOR_MATMUL_H__