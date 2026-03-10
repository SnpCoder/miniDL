#include "dataset.h"
namespace miniDL {
namespace data {
class TextDataset : public data::Dataset {
   private:
    std::string _text;
    std::unordered_map<char, int> _stoi;
    size_t _seq_len;

   public:
    TextDataset(const std::string& text, const std::unordered_map<char, int>& stoi, size_t seq_len)
        : _text(text), _stoi(stoi), _seq_len(seq_len) {}

    // 能够切分出的样本总数
    size_t size() const override {
        if (_text.size() <= _seq_len) return 0;
        return _text.size() - _seq_len;
    }

    // 核心：每次提取出一个长度为 seq_len 的片段作为 X，错位一个字符的片段作为 Y
    std::pair<Tensor, Tensor> get_item(size_t index) const override {
        TensorOptions opts;
        opts.device(Device("cpu")).requiresGrad(false);

        Tensor x = Tensor::empty(Shape({_seq_len}), opts);
        Tensor y = Tensor::empty(Shape({_seq_len}), opts);

        float* x_ptr = x.data_ptr<float>();
        float* y_ptr = y.data_ptr<float>();

        for (size_t i = 0; i < _seq_len; ++i) {
            x_ptr[i] = static_cast<float>(_stoi.at(_text[index + i]));
            y_ptr[i] = static_cast<float>(_stoi.at(_text[index + i + 1]));  // 目标：预测下一个词
        }

        return {x, y};
    }
};
}  // namespace data
}  // namespace miniDL