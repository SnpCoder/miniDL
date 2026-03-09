#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/nn/linear.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

class NNTest : public ::testing::Test {};

TEST_F(NNTest, LinearLayerInitializationAndParameters) {
    // 128 输入，256 输出，带 Bias
    nn::Linear layer(128, 256, true, Device("cpu"));

    auto params = layer.parameters();
    ASSERT_EQ(params.size(), 2);  // 应该注册了 weight 和 bias

    EXPECT_EQ(params[0].shape(), Shape({256, 128}));  // weight [out, in]
    EXPECT_EQ(params[1].shape(), Shape({256}));       // bias [out]
}

#ifdef USE_CUDA
TEST_F(NNTest, LinearLayerAutogradCUDA) {
    MINIDL_PRINT("========== 启动 CUDA nn::Linear 终极连通测试 ==========");

    nn::Linear layer(128, 256, true, Device("cuda:0"));

    // 虚拟一个 Batch = 32 的输入数据
    Tensor x =
        Tensor::randn(Shape({32, 128}), 0.0f, 1.0f, miniDL::device("cuda:0").requiresGrad(true));

    // 1. 前向传播 (触发 Matmul + BroadcastTo + Add)
    Tensor y = layer(x);
    EXPECT_EQ(y.shape(), Shape({32, 256}));

    // 2. 假设损失函数的梯度全部为 1.0，注入回去
    Tensor grad_out = Tensor::ones(Shape({32, 256}), miniDL::device("cuda:0"));
    y.impl()->set_grad(grad_out.shared_impl());

    // 3. 反向传播引擎启动！
    y.backward();

    auto params = layer.parameters();

    // 4. 严苛断言：检查权重和偏置是否都获得了物理梯度内存
    ASSERT_TRUE(params[0].grad().defined());  // weight 梯度
    ASSERT_TRUE(params[1].grad().defined());  // bias 梯度
    ASSERT_TRUE(x.grad().defined());          // 输入 x 也必须有梯度

    // 检查 bias 梯度的数学正确性 (32 行全是 1.0 缩减求和，所以必定等于 32.0)
    Tensor bias_grad_cpu = params[1].grad().to(Device("cpu"));
    EXPECT_FLOAT_EQ(bias_grad_cpu.data_ptr<float>()[0], 32.0f);
    EXPECT_FLOAT_EQ(bias_grad_cpu.data_ptr<float>()[255], 32.0f);

    MINIDL_PRINT("========== nn::Linear 连通测试完美通过！ ==========");
}

TEST_F(NNTest, ModuleDeviceTransfer) {
    nn::Linear layer(10, 5, true, Device("cpu"));

    // 将整个网络层转移到 GPU
    layer.to(Device("cuda:0"));

    auto params = layer.parameters();
    EXPECT_TRUE(params[0].device().isCuda());
    EXPECT_TRUE(params[1].device().isCuda());
}
#endif