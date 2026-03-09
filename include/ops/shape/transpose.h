#ifndef __MINIDL_OPERATOR_TRANSPOSE_H__
#define __MINIDL_OPERATOR_TRANSPOSE_H__
#include "../operator.h"

namespace miniDL {

// PyTorch 对标：TransposeBackward0 节点
class TransposeOp : public Operator {
   public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    static Tensor apply(const Tensor& a);
};

}  // namespace miniDL

#endif  // __MINIDL_OPERATOR_TRANSPOSE_H__