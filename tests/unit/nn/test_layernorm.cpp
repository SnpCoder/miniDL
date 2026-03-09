#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/nn/layernorm.h"

using namespace miniDL;

class LayerNormTest : public ::testing::Test {};

// ============================================================================
// 测试 1: 纯数学验证 (前向与反向传播的绝对精度测试)
// ============================================================================
void check_layernorm_math(Device device) {
    auto opt = miniDL::device(device).requiresGrad(true);

    // 构造一个 1行3列 的张量: [1.0, 2.0, 3.0]
    Tensor x_cpu               = Tensor::empty(Shape({1, 3}), miniDL::device("cpu"));
    x_cpu.data_ptr<float>()[0] = 1.0f;
    x_cpu.data_ptr<float>()[1] = 2.0f;
    x_cpu.data_ptr<float>()[2] = 3.0f;
    Tensor X                   = x_cpu.to(device);
    X.impl()->set_requires_grad(true);

    // 构造 weight=[1,1,1] 和 bias=[0,0,0]
    Tensor weight = Tensor::ones(Shape({3}), opt);
    Tensor bias   = Tensor::zeros(Shape({3}), opt);

    // 前向传播
    float eps = 0.0f;  // 为了手工推导方便，设为 0
    Tensor Y  = LayerNormOp::apply(X, weight, bias, Shape({3}), eps);

    // 【前向数学验证】
    // 均值 = 2.0, 方差 = 2/3, 标准差 = sqrt(2/3) = 0.8165
    // 归一化后: [-1.2247, 0.0, 1.2247]
    Tensor y_cpu = Y.to(Device("cpu"));
    float* y_ptr = y_cpu.data_ptr<float>();
    EXPECT_NEAR(y_ptr[0], -1.224744f, 1e-4);
    EXPECT_NEAR(y_ptr[1], 0.0f, 1e-4);
    EXPECT_NEAR(y_ptr[2], 1.224744f, 1e-4);

    // 【反向数学验证】
    // 注入梯度 dY = [1.0, 2.0, 0.0]
    Tensor dy_cpu               = Tensor::empty(Shape({1, 3}), miniDL::device("cpu"));
    dy_cpu.data_ptr<float>()[0] = 1.0f;
    dy_cpu.data_ptr<float>()[1] = 2.0f;
    dy_cpu.data_ptr<float>()[2] = 0.0f;
    Tensor dY                   = dy_cpu.to(device);

    Y.impl()->set_grad(dY.shared_impl());
    Y.backward();

    // 验证 dWeight = dY * X_hat
    // [-1.2247*1, 0*2, 1.2247*0] = [-1.2247, 0, 0]
    Tensor dw_cpu = weight.grad().to(Device("cpu"));
    EXPECT_NEAR(dw_cpu.data_ptr<float>()[0], -1.224744f, 1e-4);
    EXPECT_NEAR(dw_cpu.data_ptr<float>()[1], 0.0f, 1e-4);
    EXPECT_NEAR(dw_cpu.data_ptr<float>()[2], 0.0f, 1e-4);

    // 验证 dBias = sum(dY) (因为只有一行，直接等于 dY)
    Tensor db_cpu = bias.grad().to(Device("cpu"));
    EXPECT_FLOAT_EQ(db_cpu.data_ptr<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(db_cpu.data_ptr<float>()[1], 2.0f);
    EXPECT_FLOAT_EQ(db_cpu.data_ptr<float>()[2], 0.0f);

    // 验证 dX 的复杂求导
    // 工业界的隐藏彩蛋：LayerNorm 输入的梯度，其同一行之和永远严格等于 0！
    Tensor dx_cpu = X.grad().to(Device("cpu"));
    float dx0     = dx_cpu.data_ptr<float>()[0];
    float dx1     = dx_cpu.data_ptr<float>()[1];
    float dx2     = dx_cpu.data_ptr<float>()[2];

    // 我们算出的 dX 理论值约为: [-0.612, 1.2247, -0.612]
    EXPECT_NEAR(dx0, -0.61237f, 1e-4);
    EXPECT_NEAR(dx1, 1.22474f, 1e-4);
    EXPECT_NEAR(dx2, -0.61237f, 1e-4);
    // 零和定理：误差必须小于 1e-5
    EXPECT_NEAR(dx0 + dx1 + dx2, 0.0f, 1e-5);
}

TEST_F(LayerNormTest, MathCPU) {
    check_layernorm_math(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(LayerNormTest, MathCUDA) {
    check_layernorm_math(Device("cuda:0"));
}
#endif

// ============================================================================
// 测试 2: nn::Module 面向对象封装测试
// ============================================================================
void check_nn_layernorm(Device device) {
    // 模拟 Transformer 的输入: [Batch=2, Seq=4, Dim=8]
    auto opt = miniDL::device(device).requiresGrad(true);
    Tensor X = Tensor::uniform(Shape({2, 4, 8}), 0.0f, 1.0f, opt);

    // 初始化 Module
    nn::LayerNorm ln(Shape({8}), 1e-5, device);

    // 检查参数是否被正确注册
    auto params = ln.parameters();
    EXPECT_EQ(params.size(), 2);  // weight 和 bias

    // 运行前向与反向
    Tensor Y = ln(X);
    EXPECT_EQ(Y.shape().size(), 3);

    Tensor dY = Tensor::ones(Shape({2, 4, 8}), opt);
    Y.impl()->set_grad(dY.shared_impl());
    Y.backward();

    // 验证权重是否成功捕获梯度
    EXPECT_TRUE(ln.weight.grad().defined());
    EXPECT_TRUE(ln.bias.grad().defined());

    // bias 的梯度应该是 dY 在 Batch 和 Seq 维度上的累加
    // 因为 dY 全是 1，且有 2*4=8 行，所以 bias 的梯度每个元素都必须等于 8.0！
    Tensor db_cpu = ln.bias.grad().to(Device("cpu"));
    EXPECT_FLOAT_EQ(db_cpu.data_ptr<float>()[0], 8.0f);
}

TEST_F(LayerNormTest, ModuleCPU) {
    check_nn_layernorm(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(LayerNormTest, ModuleCUDA) {
    check_nn_layernorm(Device("cuda:0"));
}
#endif