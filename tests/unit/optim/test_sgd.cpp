#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/optim/sgd.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

class SGDTest : public ::testing::Test {};

// ============================================================================
// 测试 1: 纯享版 SGD (无动量，无衰减)
// 公式: W = W - lr * grad
// ============================================================================
void check_basic_sgd(Device device) {
    auto opt = miniDL::device(device).requiresGrad(true);
    Tensor w = Tensor::ones(Shape({2}), opt);  // W 初始化为 1.0

    // 手工注入梯度 2.0
    Tensor grad = Tensor::ones(Shape({2}), miniDL::device(device));
    grad += 1.0f;
    w.impl()->set_grad(grad.shared_impl());

    // 创建 SGD，学习率 0.1
    optim::SGD optimizer({w}, 0.1f);
    optimizer.step();

    // 期待结果: W = 1.0 - 0.1 * 2.0 = 0.8
    Tensor w_cpu = w.to(Device("cpu"));
    EXPECT_FLOAT_EQ(w_cpu.data_ptr<float>()[0], 0.8f);
    EXPECT_FLOAT_EQ(w_cpu.data_ptr<float>()[1], 0.8f);

    // 测试 zero_grad
    optimizer.zero_grad();
    EXPECT_FALSE(w.grad().defined());  // 梯度应当被彻底清空
}

TEST_F(SGDTest, BasicCPU) {
    check_basic_sgd(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(SGDTest, BasicCUDA) {
    check_basic_sgd(Device("cuda:0"));
}
#endif

// ============================================================================
// 测试 2: SGD with Weight Decay (L2 正则)
// 公式: g = grad + wd * W;  W = W - lr * g
// ============================================================================
void check_sgd_weight_decay(Device device) {
    auto opt = miniDL::device(device).requiresGrad(true);
    Tensor w = Tensor::ones(Shape({1}), opt);  // W = 1.0

    Tensor grad = Tensor::ones(Shape({1}), miniDL::device(device));
    grad += 1.0f;  // grad = 2.0
    w.impl()->set_grad(grad.shared_impl());

    // lr = 0.1, momentum = 0.0, weight_decay = 0.5
    optim::SGD optimizer({w}, 0.1f, 0.0f, 0.5f);
    optimizer.step();

    // 手工推演:
    // g = 2.0 + 0.5 * 1.0 = 2.5
    // W = 1.0 - 0.1 * 2.5 = 0.75
    Tensor w_cpu = w.to(Device("cpu"));
    EXPECT_FLOAT_EQ(w_cpu.data_ptr<float>()[0], 0.75f);
}

TEST_F(SGDTest, WeightDecayCPU) {
    check_sgd_weight_decay(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(SGDTest, WeightDecayCUDA) {
    check_sgd_weight_decay(Device("cuda:0"));
}
#endif

// ============================================================================
// 测试 3: SGD with Momentum (动量累积)
// 公式: buf = buf * momentum + grad;  W = W - lr * buf
// ============================================================================
void check_sgd_momentum(Device device) {
    auto opt = miniDL::device(device).requiresGrad(true);
    Tensor w = Tensor::ones(Shape({1}), opt);  // W = 1.0

    // lr = 0.1, momentum = 0.9, weight_decay = 0.0
    optim::SGD optimizer({w}, 0.1f, 0.9f, 0.0f);

    // --- 第一步 ---
    Tensor grad1 = Tensor::ones(Shape({1}), miniDL::device(device));
    grad1 += 1.0f;  // grad = 2.0
    w.impl()->set_grad(grad1.shared_impl());

    optimizer.step();
    // 推演 1:
    // buf = 0 * 0.9 + 2.0 = 2.0
    // W = 1.0 - 0.1 * 2.0 = 0.8
    EXPECT_FLOAT_EQ(w.to(Device("cpu")).data_ptr<float>()[0], 0.8f);

    // --- 第二步 (模拟下一个 Epoch) ---
    Tensor grad2 = Tensor::ones(Shape({1}), miniDL::device(device));  // grad = 1.0
    w.impl()->set_grad(grad2.shared_impl());

    optimizer.step();
    // 推演 2:
    // buf = 2.0 * 0.9 + 1.0 = 2.8
    // W = 0.8 - 0.1 * 2.8 = 0.52
    EXPECT_FLOAT_EQ(w.to(Device("cpu")).data_ptr<float>()[0], 0.52f);
}

TEST_F(SGDTest, MomentumCPU) {
    check_sgd_momentum(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(SGDTest, MomentumCUDA) {
    check_sgd_momentum(Device("cuda:0"));
}
#endif