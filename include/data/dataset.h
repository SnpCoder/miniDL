#pragma once
#include <utility>

#include "../core/tensor.h"

namespace miniDL {
namespace data {

class Dataset {
   public:
    virtual ~Dataset() = default;

    // 返回数据集包含的总样本数
    virtual size_t size() const = 0;

    // 根据索引获取单个样本的 特征(X) 和 标签(Y)
    virtual std::pair<Tensor, Tensor> get_item(size_t index) const = 0;
};

}  // namespace data
}  // namespace miniDL