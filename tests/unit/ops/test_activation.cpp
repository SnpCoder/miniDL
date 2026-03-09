#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/nn/activation.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

class ActivationTest : public ::testing::Test {};

// ============================================================================
// 1. ReLU 测试
// ============================================================================
void check_relu(Device device) {
    auto opt     = miniDL::device(device).requiresGrad(true);
    Tensor x_cpu = Tensor::empty(Shape({4}), miniDL::device("cpu"));
    float* x_ptr = x_cpu.data_ptr<float>();
    x_ptr[0]     = -1.0f;
    x_ptr[1]     = 0.0f;
    x_ptr[2]     = 1.0f;
    x_ptr[3]     = 2.0f;

    Tensor x = x_cpu.to(device);
    x.impl()->set_requires_grad(true);

    nn::ReLU relu;
    Tensor y = relu(x);

    // 前向验证
    Tensor y_cpu = y.to(Device("cpu"));
    float* y_ptr = y_cpu.data_ptr<float>();
    EXPECT_FLOAT_EQ(y_ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(y_ptr[1], 0.0f);
    EXPECT_FLOAT_EQ(y_ptr[2], 1.0f);
    EXPECT_FLOAT_EQ(y_ptr[3], 2.0f);

    // 反向验证
    Tensor grad_out = Tensor::ones(Shape({4}), miniDL::device(device));
    y.impl()->set_grad(grad_out.shared_impl());
    y.backward();

    Tensor gx_cpu = x.grad().to(Device("cpu"));
    float* gx_ptr = gx_cpu.data_ptr<float>();
    EXPECT_FLOAT_EQ(gx_ptr[0], 0.0f);  // x < 0, 梯度为0
    EXPECT_FLOAT_EQ(gx_ptr[1], 0.0f);  // x = 0, 梯度为0
    EXPECT_FLOAT_EQ(gx_ptr[2], 1.0f);  // x > 0, 梯度为1
    EXPECT_FLOAT_EQ(gx_ptr[3], 1.0f);  // x > 0, 梯度为1
}

TEST_F(ActivationTest, ReluCPU) {
    check_relu(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(ActivationTest, ReluCUDA) {
    check_relu(Device("cuda:0"));
}
#endif

// ============================================================================
// 2. GELU 测试
// ============================================================================
void check_gelu(Device device) {
    auto opt                   = miniDL::device(device).requiresGrad(true);
    Tensor x_cpu               = Tensor::empty(Shape({2}), miniDL::device("cpu"));
    x_cpu.data_ptr<float>()[0] = 0.0f;
    x_cpu.data_ptr<float>()[1] = 1.0f;

    Tensor x = x_cpu.to(device);
    x.impl()->set_requires_grad(true);

    nn::GELU gelu;
    Tensor y = gelu(x);

    // 前向验证
    Tensor y_cpu = y.to(Device("cpu"));
    EXPECT_NEAR(y_cpu.data_ptr<float>()[0], 0.0f, 1e-4);
    EXPECT_NEAR(y_cpu.data_ptr<float>()[1], 0.8413f, 1e-3);  // GELU(1) 约等于 0.8413

    // 反向验证连通性
    Tensor grad_out = Tensor::ones(Shape({2}), miniDL::device(device));
    y.impl()->set_grad(grad_out.shared_impl());
    y.backward();
    EXPECT_TRUE(x.grad().defined());
}

TEST_F(ActivationTest, GeluCPU) {
    check_gelu(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(ActivationTest, GeluCUDA) {
    check_gelu(Device("cuda:0"));
}
#endif

// ============================================================================
// 3. Softmax 测试 (包含雅可比矩阵降维导数验证)
// ============================================================================
void check_softmax(Device device) {
    auto opt                   = miniDL::device(device).requiresGrad(true);
    Tensor x_cpu               = Tensor::empty(Shape({1, 2}), miniDL::device("cpu"));
    x_cpu.data_ptr<float>()[0] = 10.0f;  // [10, 10] 应该输出 [0.5, 0.5]
    x_cpu.data_ptr<float>()[1] = 10.0f;

    Tensor x = x_cpu.to(device);
    x.impl()->set_requires_grad(true);

    nn::Softmax softmax;
    Tensor y = softmax(x);

    // 前向验证
    Tensor y_cpu = y.to(Device("cpu"));
    EXPECT_FLOAT_EQ(y_cpu.data_ptr<float>()[0], 0.5f);
    EXPECT_FLOAT_EQ(y_cpu.data_ptr<float>()[1], 0.5f);

    // 反向验证：注入特定的梯度 dY = [1.0, 0.0]
    // 根据公式: dX = Y * (dY - sum(dY*Y))
    // sum(dY*Y) = 1*0.5 + 0*0.5 = 0.5
    // dX[0] = 0.5 * (1.0 - 0.5) = 0.25
    // dX[1] = 0.5 * (0.0 - 0.5) = -0.25
    Tensor grad_out_cpu               = Tensor::empty(Shape({1, 2}), miniDL::device("cpu"));
    grad_out_cpu.data_ptr<float>()[0] = 1.0f;
    grad_out_cpu.data_ptr<float>()[1] = 0.0f;
    Tensor grad_out                   = grad_out_cpu.to(device);

    y.impl()->set_grad(grad_out.shared_impl());
    y.backward();

    Tensor gx_cpu = x.grad().to(Device("cpu"));
    EXPECT_FLOAT_EQ(gx_cpu.data_ptr<float>()[0], 0.25f);
    EXPECT_FLOAT_EQ(gx_cpu.data_ptr<float>()[1], -0.25f);
}

TEST_F(ActivationTest, SoftmaxCPU) {
    check_softmax(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(ActivationTest, SoftmaxCUDA) {
    check_softmax(Device("cuda:0"));
}
#endif