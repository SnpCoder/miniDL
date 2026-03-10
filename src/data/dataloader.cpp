#include "../../include/data/dataloader.h"

#include <algorithm>
#include <cstring>  // for std::memcpy

#include "../../include/utils/exception.h"

namespace miniDL {
namespace data {

DataLoader::DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle, bool drop_last)
    : _dataset(dataset)
    , _batch_size(batch_size)
    , _shuffle(shuffle)
    , _drop_last(drop_last)
    , _rng(42) {
    // 初始化索引数组 [0, 1, 2, ..., size-1]
    _indices.resize(dataset.size());
    std::iota(_indices.begin(), _indices.end(), 0);
}

void DataLoader::reset() {
    if (_shuffle) { std::shuffle(_indices.begin(), _indices.end(), _rng); }
}

std::pair<Tensor, Tensor> DataLoader::collate_fn(const std::vector<size_t>& batch_indices) const {
    size_t actual_batch_size = batch_indices.size();
    if (actual_batch_size == 0) { MINIDL_THROW_RUNTIME("Cannot collate an empty batch."); }

    // 先拿第一个样本出来，探明它的形状
    auto first_item = _dataset.get_item(batch_indices[0]);
    Tensor x_sample = first_item.first;
    Tensor y_sample = first_item.second;

    // 必须确保它们目前都在 CPU 上（DataLoader 的组装必须在 CPU 内存中完成）
    if (!x_sample.device().isCpu() || !y_sample.device().isCpu()) {
        MINIDL_THROW_RUNTIME("Dataset must return CPU tensors for DataLoader collation.");
    }

    // 计算拼接后的目标形状 [Batch_size, ...]
    std::vector<size_t> x_batch_shape = {actual_batch_size};
    for (auto d : x_sample.shape().vec()) x_batch_shape.push_back(d);

    std::vector<size_t> y_batch_shape = {actual_batch_size};
    for (auto d : y_sample.shape().vec()) y_batch_shape.push_back(d);

    // 分配连续的 Batch 内存空间
    Tensor batch_x = Tensor::empty(Shape(x_batch_shape), x_sample.options());
    Tensor batch_y = Tensor::empty(Shape(y_batch_shape), y_sample.options());

    float* bx_ptr = batch_x.data_ptr<float>();
    float* by_ptr = batch_y.data_ptr<float>();

    size_t x_offset = x_sample.element_num();
    size_t y_offset = y_sample.element_num();

    // 将各个独立样本的内存拷贝到统一的 Batch 张量中
    for (size_t i = 0; i < actual_batch_size; ++i) {
        auto item = _dataset.get_item(batch_indices[i]);

        // 使用 C 语言底层的极速内存拷贝
        std::memcpy(bx_ptr + i * x_offset, item.first.data_ptr<float>(), x_offset * sizeof(float));

        std::memcpy(by_ptr + i * y_offset, item.second.data_ptr<float>(), y_offset * sizeof(float));
    }

    return {batch_x, batch_y};
}

}  // namespace data
}  // namespace miniDL