#include <getopt.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "../include/core/tensor.h"
#include "../include/data/dataloader.h"
#include "../include/data/text_dataset.h"
#include "../include/nn/embedding.h"
#include "../include/nn/linear.h"
#include "../include/nn/loss.h"
#include "../include/nn/transformer_block.h"
#include "../include/optim/adamw.h"  // 【换装】引入 AdamW
#include "../include/utils/log.h"

using namespace miniDL;

// ============================================================================
// 1. 定义我们自己的微型大语言模型 (纯正 NLP 分类架构)
// ============================================================================
class MiniatureGPT : public nn::Module {
   private:
    std::shared_ptr<nn::Embedding> token_emb;
    std::shared_ptr<nn::Embedding> pos_emb;
    std::vector<std::shared_ptr<nn::TransformerBlock>> blocks;  // multi-block
    std::shared_ptr<nn::Linear> head;

   public:
    MiniatureGPT(size_t vocab_size, size_t d_model, size_t num_heads, size_t num_layers,
                 size_t max_seq_len, Device device) {
        token_emb = std::make_shared<nn::Embedding>(vocab_size, d_model, device);
        pos_emb   = std::make_shared<nn::Embedding>(max_seq_len, d_model, device);

        // 循环堆叠多层 Transformer Block
        for (size_t i = 0; i < num_layers; ++i) {
            auto block = std::make_shared<nn::TransformerBlock>(d_model, num_heads, 4, device);
            blocks.push_back(block);
            // 必须分别注册，否则优化器找不到它们！
            register_module("block_" + std::to_string(i), block);
        }

        head = std::make_shared<nn::Linear>(d_model, vocab_size, true, device);

        register_module("token_emb", token_emb);
        register_module("pos_emb", pos_emb);
        register_module("head", head);
    }

    // 现在 forward 需要两个输入：词的 ID，以及它们对应的位置 ID
    Tensor forward(const Tensor& x, const Tensor& pos) {
        size_t B = x.shape()[0];
        size_t S = x.shape()[1];

        // 1. 获取 Token 向量和 Position 向量
        Tensor tok_h = (*token_emb)(x);
        Tensor pos_h = (*pos_emb)(pos);

        // 2. 将它们直接相加！(你的 Tensor operator+ 已经支持了！)
        Tensor h = tok_h + pos_h;

        // 让数据逐层穿过所有的 Transformer Block
        for (auto& block : blocks) { h = (*block)(h); }

        Tensor h_flat = h.reshape(Shape({B * S, h.shape()[2]}));
        Tensor out    = (*head)(h_flat);

        return out;
    }
};

void usage(const char* program_name) {
    MINIDL_PRINT("Usage: {} [--cpu] [-d device_id] [-h]", program_name);
}

int main(int argc, char** argv) {
    Device target_device("cuda:0");

    // ========================================================================
    // 3. 准备真实世界的数据
    // ========================================================================
    std::ifstream file("../tests/example/words.txt");
    if (!file.is_open()) {
        MINIDL_PRINT("❌ 请在当前目录创建一个 words.txt 并放入一段英文文本！");
        return -1;
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<char> chars;
    for (char c : text) {
        if (std::find(chars.begin(), chars.end(), c) == chars.end()) chars.push_back(c);
    }
    std::sort(chars.begin(), chars.end());

    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;
    for (size_t i = 0; i < chars.size(); ++i) {
        stoi[chars[i]] = i;
        itos[i]        = chars[i];
    }
    size_t vocab_size = chars.size();
    MINIDL_PRINT("📚 词表构建完毕！总字数: {}, Vocab Size: {}", text.size(), vocab_size);

    // ========================================================================
    // 4. 配置超参数与实例化 DataLoader
    // ========================================================================
    size_t seq_len      = 64;
    size_t d_model      = 256;
    size_t num_heads    = 8;
    size_t num_layers   = 4;
    size_t max_seq_len  = 256;
    size_t batch_size   = 64;
    float learning_rate = 0.0003f;
    int epochs          = 200;

    // 实例化数据集和高阶迭代器
    data::TextDataset dataset(text, stoi, seq_len);
    // 丢弃最后一个不完整的 batch，保证形状严格一致，对矩阵乘法极其友好！
    data::DataLoader dataloader(dataset, batch_size, /*shuffle=*/true, /*drop_last=*/true);

    MiniatureGPT model(vocab_size, d_model, num_heads, num_layers, max_seq_len, target_device);
    nn::CrossEntropyLoss criterion;
    optim::AdamW optimizer(model.parameters(), learning_rate);

    // ========================================================================
    // 5. 极致优雅的 PyTorch 风格训练循环
    // ========================================================================
    MINIDL_PRINT("🚀 DataLoader 已挂载，模型开始贪婪吞噬文本...");
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int step         = 0;

        // 【看！这就是 DataLoader 带来的极其优美的范围 for 循环！】
        for (auto batch : dataloader) {
            Tensor x_cpu           = batch.first;
            Tensor y_cpu           = batch.second;
            size_t curr_batch_size = x_cpu.shape()[0];

            // 动态生成绝对位置编码 [Batch, Seq]
            Tensor pos_cpu = Tensor::empty(Shape({curr_batch_size, seq_len}),
                                           miniDL::device("cpu").requiresGrad(false));
            float* pos_ptr = pos_cpu.data_ptr<float>();
            for (size_t b = 0; b < curr_batch_size; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    pos_ptr[b * seq_len + s] = static_cast<float>(s);
                }
            }

            // 数据飞往 GPU
            Tensor x   = x_cpu.to(target_device);
            Tensor pos = pos_cpu.to(target_device);
            Tensor y   = y_cpu.to(target_device);

            // 前向与反向传播
            optimizer.zero_grad();
            Tensor logits = model.forward(x, pos);

            // 注意：交叉熵要求 target 是 1D 的，所以我们需要把 y_batch 压扁
            Tensor y_flat = y.reshape(Shape({curr_batch_size * seq_len}));
            Tensor loss   = criterion(logits, y_flat);

            Tensor grad_out = Tensor::ones(Shape({1}), miniDL::device(target_device));
            loss.impl()->set_grad(grad_out.shared_impl());
            loss.backward();
            optimizer.step();

            epoch_loss += loss.to(Device("cpu")).data_ptr<float>()[0];
            step++;
        }

        // 打印 Epoch 均值 Loss
        if (epoch % 10 == 0 || epoch == epochs) {
            float avg_loss = epoch_loss / step;
            MINIDL_PRINT("Epoch {:3d} | Avg Loss: {:.4f} | PPL: {:.2f}", epoch, avg_loss,
                         std::exp(avg_loss));
        }
    }

    // ========================================================================
    // 6. 魔法时刻：自回归推理生成
    // ========================================================================
    MINIDL_PRINT("✨ 训练完毕！正在将控制权交给 AI...");
    MINIDL_PRINT("\n================= AI 生成的文本 =================\n");

    std::string prompt_str = "Here's to";
    MINIDL_PRINT("{}", prompt_str);

    std::vector<int> context;
    for (char c : prompt_str) context.push_back(stoi[c]);

    TensorOptions cpu_opts;
    cpu_opts.device(Device("cpu")).requiresGrad(false);

    int generate_length = 200;

    // 【新增】：创造力温度 (0.0 ~ 1.0)
    // 越接近 0 越严谨死板，越接近 1 越天马行空
    float temperature = 0.4f;

    for (int step = 0; step < generate_length; ++step) {
        size_t current_len = context.size();
        size_t infer_len   = std::min(current_len, seq_len);

        Tensor infer_x_cpu   = Tensor::empty(Shape({1, infer_len}), cpu_opts);
        Tensor infer_pos_cpu = Tensor::empty(Shape({1, infer_len}), cpu_opts);

        for (size_t i = 0; i < infer_len; ++i) {
            infer_x_cpu.data_ptr<float>()[i] =
                static_cast<float>(context[current_len - infer_len + i]);
            infer_pos_cpu.data_ptr<float>()[i] = static_cast<float>(i);
        }

        Tensor infer_x   = infer_x_cpu.to(target_device);
        Tensor infer_pos = infer_pos_cpu.to(target_device);

        Tensor logits = model.forward(infer_x, infer_pos);

        // ====================================================================
        // 【修改 2：内存泄露克星】使用 Fake Backward 清理计算图垃圾！
        // ====================================================================
        Tensor dummy_grad = Tensor::zeros(logits.shape(), miniDL::device(target_device));
        logits.impl()->set_grad(dummy_grad.shared_impl());
        logits.backward();  // 触发析构和缓存清理，绝不调用 optimizer.step()！

        Tensor logits_cpu        = logits.to(Device("cpu"));
        float* logits_ptr        = logits_cpu.data_ptr<float>();
        size_t last_token_offset = (infer_len - 1) * vocab_size;

        // ====================================================================
        // 【修改 3：轮盘赌采样 (Multinomial Sampling)】赋予 AI 真正的灵魂
        // ====================================================================
        float max_logit = -1e9f;
        for (size_t v = 0; v < vocab_size; ++v) {
            max_logit = std::max(max_logit, logits_ptr[last_token_offset + v]);
        }

        float sum_prob = 0.0f;
        std::vector<float> probs(vocab_size);
        for (size_t v = 0; v < vocab_size; ++v) {
            // Temperature 缩放
            float p  = std::exp((logits_ptr[last_token_offset + v] - max_logit) / temperature);
            probs[v] = p;
            sum_prob += p;
        }

        // 归一化并生成 0~1 的随机数
        float r           = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float cumulative  = 0.0f;
        int next_token_id = 0;

        // 轮盘赌抽奖
        for (size_t v = 0; v < vocab_size; ++v) {
            probs[v] /= sum_prob;
            cumulative += probs[v];
            if (r <= cumulative) {
                next_token_id = v;
                break;
            }
        }

        context.push_back(next_token_id);
        std::cout << itos[next_token_id] << std::flush;
    }

    MINIDL_PRINT("\n================= AI 生成的文本 =================\n");
    return 0;
}