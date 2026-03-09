#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/nn/embedding.h"

using namespace miniDL;

class EmbeddingTest : public ::testing::Test {};

// ============================================================================
// 测试 1: 纯数学验证 (前向查表与反向梯度累加)
// ============================================================================
void check_embedding_math(Device device) {
    auto opt = miniDL::device(device).requiresGrad(true);

    // 1. 构造词表权重 (Vocab Size = 5, Embed Dim = 3)
    Tensor weight_cpu = Tensor::empty(Shape({5, 3}), miniDL::device("cpu"));
    float* w_ptr      = weight_cpu.data_ptr<float>();
    // 第 0 行: 0, 0, 0
    w_ptr[0] = 0;
    w_ptr[1] = 0;
    w_ptr[2] = 0;
    // 第 1 行: 1, 1, 1
    w_ptr[3] = 1;
    w_ptr[4] = 1;
    w_ptr[5] = 1;
    // 第 2 行: 2, 2, 2
    w_ptr[6] = 2;
    w_ptr[7] = 2;
    w_ptr[8] = 2;
    // 第 3 行: 3, 3, 3
    w_ptr[9]  = 3;
    w_ptr[10] = 3;
    w_ptr[11] = 3;
    // 第 4 行: 4, 4, 4
    w_ptr[12] = 4;
    w_ptr[13] = 4;
    w_ptr[14] = 4;

    Tensor weight = weight_cpu.to(device);
    weight.impl()->set_requires_grad(true);

    // 2. 构造输入的索引序列 [Batch=2, Seq=2]
    // 故意让词 ID '2' 出现两次，测试梯度累加！
    Tensor indices_cpu = Tensor::empty(Shape({2, 2}), miniDL::device("cpu"));
    float* idx_ptr     = indices_cpu.data_ptr<float>();
    idx_ptr[0]         = 2.0f;  // 查词表第 2 行
    idx_ptr[1]         = 0.0f;  // 查词表第 0 行
    idx_ptr[2]         = 2.0f;  // 又查词表第 2 行 (重点！)
    idx_ptr[3]         = 4.0f;  // 查词表第 4 行
    Tensor indices     = indices_cpu.to(device);

    // 3. 执行前向传播
    Tensor out = EmbeddingOp::apply(indices, weight);

    // 验证前向输出形状 [2, 2, 3]
    EXPECT_EQ(out.shape().size(), 3);
    EXPECT_EQ(out.shape()[0], 2);
    EXPECT_EQ(out.shape()[1], 2);
    EXPECT_EQ(out.shape()[2], 3);

    // 验证前向查表结果是否准确
    Tensor out_cpu = out.to(Device("cpu"));
    float* o_ptr   = out_cpu.data_ptr<float>();
    // out[0, 0, :] 应该是 weight[2] -> 2, 2, 2
    EXPECT_FLOAT_EQ(o_ptr[0], 2.0f);
    // out[0, 1, :] 应该是 weight[0] -> 0, 0, 0
    EXPECT_FLOAT_EQ(o_ptr[3], 0.0f);
    // out[1, 0, :] 应该是 weight[2] -> 2, 2, 2
    EXPECT_FLOAT_EQ(o_ptr[6], 2.0f);
    // out[1, 1, :] 应该是 weight[4] -> 4, 4, 4
    EXPECT_FLOAT_EQ(o_ptr[9], 4.0f);

    // 4. 反向传播梯度累加测试 (核心护城河！)
    // 给每一维注入 1.0 的梯度
    Tensor grad_out = Tensor::ones(Shape({2, 2, 3}), miniDL::device(device));
    out.impl()->set_grad(grad_out.shared_impl());
    out.backward();

    Tensor grad_w_cpu = weight.grad().to(Device("cpu"));
    float* gw_ptr     = grad_w_cpu.data_ptr<float>();

    // 验证梯度：
    // 词 ID '0' 出现 1 次 -> 梯度全是 1.0
    EXPECT_FLOAT_EQ(gw_ptr[0], 1.0f);
    // 词 ID '1' 出现 0 次 -> 梯度全是 0.0
    EXPECT_FLOAT_EQ(gw_ptr[3], 0.0f);
    // 词 ID '2' 出现 2 次 -> 梯度必须累加到 2.0！(检验 atomicAdd)
    EXPECT_FLOAT_EQ(gw_ptr[6], 2.0f);
    EXPECT_FLOAT_EQ(gw_ptr[7], 2.0f);
    EXPECT_FLOAT_EQ(gw_ptr[8], 2.0f);
    // 词 ID '3' 出现 0 次 -> 梯度全是 0.0
    EXPECT_FLOAT_EQ(gw_ptr[9], 0.0f);
    // 词 ID '4' 出现 1 次 -> 梯度全是 1.0
    EXPECT_FLOAT_EQ(gw_ptr[12], 1.0f);
}

TEST_F(EmbeddingTest, MathCPU) {
    check_embedding_math(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(EmbeddingTest, MathCUDA) {
    check_embedding_math(Device("cuda:0"));
}
#endif

// ============================================================================
// 测试 2: nn::Module 面向对象封装与越界保护测试
// ============================================================================
void check_nn_embedding(Device device) {
    auto opt = miniDL::device(device).requiresGrad(false);

    // 初始化 Module: 词表大小 10, 维度 16
    nn::Embedding emb(10, 16, device);

    auto params = emb.parameters();
    EXPECT_EQ(params.size(), 1);  // 只有一个 weight 矩阵
    EXPECT_EQ(params[0].shape()[0], 10);
    EXPECT_EQ(params[0].shape()[1], 16);

    // 构造测试索引 (包含一个非法的越界索引 99！)
    Tensor indices_cpu               = Tensor::empty(Shape({3}), miniDL::device("cpu"));
    indices_cpu.data_ptr<float>()[0] = 1.0f;
    indices_cpu.data_ptr<float>()[1] = 99.0f;  // 越界词
    indices_cpu.data_ptr<float>()[2] = 5.0f;
    Tensor indices                   = indices_cpu.to(device);

    Tensor out = emb(indices);
    EXPECT_EQ(out.shape().size(), 2);
    EXPECT_EQ(out.shape()[0], 3);
    EXPECT_EQ(out.shape()[1], 16);

    // 验证越界词的输出必须被保护为全 0，不能导致段错误！
    Tensor out_cpu = out.to(Device("cpu"));
    float* o_ptr   = out_cpu.data_ptr<float>();

    // 检查第 1 个词（对应原句第 1 个 token，也就是 99.0f 查出来的结果）
    // 起始内存偏移量是 1 * 16 = 16
    for (int i = 0; i < 16; ++i) { EXPECT_FLOAT_EQ(o_ptr[16 + i], 0.0f); }
}

TEST_F(EmbeddingTest, ModuleCPU) {
    check_nn_embedding(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(EmbeddingTest, ModuleCUDA) {
    check_nn_embedding(Device("cuda:0"));
}
#endif