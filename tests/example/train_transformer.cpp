#include <getopt.h>

#include <cstdlib>
#include <random>
#include <vector>

#include "../include/core/tensor.h"
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
    std::shared_ptr<nn::Embedding> emb;
    std::shared_ptr<nn::TransformerBlock> block;
    std::shared_ptr<nn::Linear> head;

   public:
    MiniatureGPT(size_t vocab_size, size_t d_model, size_t num_heads, Device device) {
        emb   = std::make_shared<nn::Embedding>(vocab_size, d_model, device);
        block = std::make_shared<nn::TransformerBlock>(d_model, num_heads, 4, device);

        // 【核心修改 1】：输出维度不再是 1，而是 vocab_size！
        // 模型需要输出词表中每一个词的概率 (Logits)
        head = std::make_shared<nn::Linear>(d_model, vocab_size, true, device);

        register_module("emb", emb);
        register_module("block", block);
        register_module("head", head);
    }

    Tensor forward(const Tensor& x) {
        size_t B = x.shape()[0];
        size_t S = x.shape()[1];

        Tensor h = (*emb)(x);
        h        = (*block)(h);

        Tensor h_flat = h.reshape(Shape({B * S, h.shape()[2]}));

        // out 形状: [Batch * Seq, Vocab_Size]
        Tensor out = (*head)(h_flat);

        return out;
    }
};

void usage(const char* program_name) {
    MINIDL_PRINT("Usage: {} [--cpu] [-d device_id] [-h]", program_name);
}

int main(int argc, char** argv) {
    int device_id                       = -2;
    static struct option long_options[] = {{"cpu", no_argument, 0, 'c'},
                                           {"device", required_argument, 0, 'd'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int opt_idx = 0;
    int c;
    while ((c = getopt_long(argc, argv, "cd:h", long_options, &opt_idx)) != -1) {
        if (c == 'c')
            device_id = -1;
        else if (c == 'd')
            device_id = std::atoi(optarg);
        else {
            usage(argv[0]);
            std::exit(0);
        }
    }

    Device target_device(Device("cpu"));
    if (device_id == -2) {
#ifdef USE_CUDA
        device_id = 0;
#else
        device_id = -1;
#endif
    }
    if (device_id >= 0) target_device = Device(std::string("cuda:") + std::to_string(device_id));

    // ========================================================================
    // 超参数设定
    // ========================================================================
    size_t batch_size   = 4;
    size_t seq_len      = 8;
    size_t vocab_size   = 20;  // 词表大小
    size_t d_model      = 64;
    size_t num_heads    = 4;
    int epochs          = 100;
    float learning_rate = 0.001f;  // AdamW 的绝佳学习率

    MINIDL_PRINT("====================================================");
    MINIDL_PRINT("🚀 miniDL 终极首秀：GPT 语言模型点火！");
    MINIDL_PRINT("📦 架构: Vocab={}, Dim={}, Heads={}", vocab_size, d_model, num_heads);
    MINIDL_PRINT("💻 设备: {}", target_device.to_string());
    MINIDL_PRINT("====================================================");

    // ========================================================================
    // 准备 NLP 训练数据
    // ========================================================================
    TensorOptions cpu_opts;
    cpu_opts.device(Device("cpu")).requiresGrad(false);

    Tensor x_cpu = Tensor::empty(Shape({batch_size, seq_len}), cpu_opts);

    // 【核心修改 2】：CrossEntropy 的 target 必须是 1D 的类别索引！
    Tensor y_cpu = Tensor::empty(Shape({batch_size * seq_len}), cpu_opts);

    float* x_ptr = x_cpu.data_ptr<float>();
    float* y_ptr = y_cpu.data_ptr<float>();
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist_vocab(0, vocab_size - 1);

    for (size_t i = 0; i < batch_size * seq_len; ++i) {
        int token_id = dist_vocab(gen);
        x_ptr[i]     = static_cast<float>(token_id);

        // 模拟 NLP 任务：预测下一个词的 ID。
        // 这里简单造一个规律：让它预测词表里的下一个词 (token_id + 1)
        y_ptr[i] = static_cast<float>((token_id + 1) % vocab_size);
    }

    Tensor x        = x_cpu.to(target_device);
    Tensor target_y = y_cpu.to(target_device);

    // ========================================================================
    // 实例化模型、交叉熵损失与 AdamW 优化器
    // ========================================================================
    MiniatureGPT model(vocab_size, d_model, num_heads, target_device);

    // 【核心修改 3】：换上真正的分类损失函数！
    nn::CrossEntropyLoss criterion;

    // 【核心修改 4】：换上大模型重型引擎 AdamW！
    optim::AdamW optimizer(model.parameters(), learning_rate);

    MINIDL_PRINT("⚙️ 模型参数总量查收: 共 {} 个可学习张量", model.parameters().size());

    // ========================================================================
    // 训练主循环
    // ========================================================================
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        optimizer.zero_grad();

        // 此时的 y_pred 是模型给出的 [Batch*Seq, Vocab_Size] 的 Logits 分数
        Tensor y_pred = model.forward(x);

        // CUDA 极速交叉熵计算
        Tensor loss = criterion(y_pred, target_y);

        Tensor grad_out = Tensor::ones(Shape({1}), miniDL::device(target_device));
        loss.impl()->set_grad(grad_out.shared_impl());
        loss.backward();

        optimizer.step();

        if (epoch % 10 == 0 || epoch == epochs) {
            Tensor loss_cpu = loss.to(Device("cpu"));
            float loss_val  = loss_cpu.data_ptr<float>()[0];

            // 计算一下困惑度 (Perplexity, 大模型核心评估指标)
            float ppl = std::exp(loss_val);

            MINIDL_PRINT("Epoch {:3d} / {} | CrossEntropy Loss: {:.6f} | PPL: {:.2f}", epoch,
                         epochs, loss_val, ppl);
        }
    }

    MINIDL_PRINT("🎉 见证奇迹：GPT 成功收敛！你已经掌握了大语言模型的最核心科技！");
    return 0;
}