#pragma once
#include <numeric>
#include <random>
#include <vector>

#include "dataset.h"

namespace miniDL {
namespace data {

class DataLoader {
   private:
    const Dataset& _dataset;
    size_t _batch_size;
    bool _shuffle;
    bool _drop_last;
    std::vector<size_t> _indices;  // 用于存放打乱后的索引
    std::mt19937 _rng;             // 随机数生成器

    // 核心的拼装函数：将单个的样本拼成一个 Batch
    std::pair<Tensor, Tensor> collate_fn(const std::vector<size_t>& batch_indices) const;

   public:
    DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle = true,
               bool drop_last = false);

    // 每次迭代前，重新打乱索引
    void reset();

    // ========================================================================
    // C++ 迭代器支持 (为了能使用 for(auto batch : dataloader) 语法)
    // ========================================================================
    class Iterator {
       private:
        DataLoader* _loader;
        size_t _current_idx;

       public:
        Iterator(DataLoader* loader, size_t start_idx) : _loader(loader), _current_idx(start_idx) {}

        // 解引用：获取当前的 Batch
        std::pair<Tensor, Tensor> operator*() const {
            std::vector<size_t> batch_indices;
            size_t end_idx =
                std::min(_current_idx + _loader->_batch_size, _loader->_indices.size());
            for (size_t i = _current_idx; i < end_idx; ++i) {
                batch_indices.push_back(_loader->_indices[i]);
            }
            return _loader->collate_fn(batch_indices);
        }

        // 步进：前进到下一个 Batch
        Iterator& operator++() {
            _current_idx += _loader->_batch_size;
            return *this;
        }

        // 判断是否结束
        bool operator!=(const Iterator& other) const {
            // 如果开启了 drop_last，不足一个 batch 的数据将被丢弃
            if (_loader->_drop_last) {
                return _current_idx + _loader->_batch_size <= _loader->_indices.size() &&
                       _current_idx != other._current_idx;
            }
            return _current_idx < _loader->_indices.size() && _current_idx != other._current_idx;
        }
    };

    Iterator begin() {
        reset();  // 每次 for 循环开始时自动洗牌
        return Iterator(this, 0);
    }

    Iterator end() { return Iterator(this, _indices.size()); }
};

}  // namespace data
}  // namespace miniDL