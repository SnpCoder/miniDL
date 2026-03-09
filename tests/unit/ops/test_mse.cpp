#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/nn/loss.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

class LossTest : public ::testing::Test {};

// ============================================================================
// 核心数学验证：MSE Loss 前向与反向
// ============================================================================
void check_mse_math(Device device) {
    auto opt = miniDL::device(device).requiresGrad(true);

    // 1. 构造预测值 pred: [1.0, 2.0, 3.0, 4.0]
    Tensor pred_cpu = Tensor::empty(Shape({4}), miniDL::device("cpu"));
    float* p_ptr    = pred_cpu.data_ptr<float>();
    p_ptr[0]        = 1.0f;
    p_ptr[1]        = 2.0f;
    p_ptr[2]        = 3.0f;
    p_ptr[3]        = 4.0f;
    Tensor pred     = pred_cpu.to(device);
    pred.impl()->set_requires_grad(true);

    // 2. 构造真实值 target: [1.0, 1.0, 5.0, 6.0] (不需要梯度)
    Tensor target_cpu = Tensor::empty(Shape({4}), miniDL::device("cpu"));
    float* t_ptr      = target_cpu.data_ptr<float>();
    t_ptr[0]          = 1.0f;
    t_ptr[1]          = 1.0f;
    t_ptr[2]          = 5.0f;
    t_ptr[3]          = 6.0f;
    Tensor target     = target_cpu.to(device);

    // 3. 实例化损失函数并前向传播
    nn::MSELoss criterion;
    Tensor loss = criterion(pred, target);

    // 4. 验证前向传播 (MSE 数学计算)
    // Diff: [0.0, 1.0, -2.0, -2.0]
    // Diff^2: [0.0, 1.0, 4.0, 4.0] -> Sum = 9.0
    // MSE = 9.0 / 4 = 2.25
    Tensor loss_cpu = loss.to(Device("cpu"));
    EXPECT_FLOAT_EQ(loss_cpu.data_ptr<float>()[0], 2.25f);

    // 5. 反向传播
    // 在 PyTorch 中，loss.backward() 会隐式地将 loss 的初始梯度设为 1.0
    Tensor grad_out = Tensor::ones(Shape({1}), miniDL::device(device));
    loss.impl()->set_grad(grad_out.shared_impl());
    loss.backward();

    // 6. 验证反向传播梯度 (公式: 2/N * (pred - target))
    // 2/4 * Diff = 0.5 * [0.0, 1.0, -2.0, -2.0] = [0.0, 0.5, -1.0, -1.0]
    Tensor grad_pred_cpu = pred.grad().to(Device("cpu"));
    float* gp_ptr        = grad_pred_cpu.data_ptr<float>();
    EXPECT_FLOAT_EQ(gp_ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(gp_ptr[1], 0.5f);
    EXPECT_FLOAT_EQ(gp_ptr[2], -1.0f);
    EXPECT_FLOAT_EQ(gp_ptr[3], -1.0f);
}

TEST_F(LossTest, MseMathCPU) {
    check_mse_math(Device("cpu"));
}

#ifdef USE_CUDA
TEST_F(LossTest, MseMathCUDA) {
    check_mse_math(Device("cuda:0"));
}
#endif