#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

class ShapeTest : public ::testing::Test {};

// ============================================================================
// 测试 1: Reshape 零拷贝与反向传播
// ============================================================================
void check_reshape(Device device) {
    auto opt = miniDL::device(device).requiresGrad(true);
    // 创建一个 [2, 3] 的张量
    Tensor x = Tensor::empty(Shape({2, 3}), opt);

    // 变形为 [3, 2]
    Tensor y = x.reshape(Shape({3, 2}));

    // 1. 验证形状变化
    EXPECT_EQ(y.shape()[0], 3);
    EXPECT_EQ(y.shape()[1], 2);

    // 2. 【硬核验证】：验证是否真的是零拷贝！(底层物理指针必须完全相同)
    EXPECT_EQ(x.data_ptr<float>(), y.data_ptr<float>());

    // 3. 反向传播连通性验证
    Tensor grad_out = Tensor::ones(Shape({3, 2}), miniDL::device(device));
    y.impl()->set_grad(grad_out.shared_impl());
    y.backward();

    // 传回给 X 的梯度必须变回 [2, 3]
    EXPECT_TRUE(x.grad().defined());
    EXPECT_EQ(x.grad().shape()[0], 2);
    EXPECT_EQ(x.grad().shape()[1], 3);
}

TEST_F(ShapeTest, ReshapeCPU) {
    check_reshape(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(ShapeTest, ReshapeCUDA) {
    check_reshape(Device("cuda:0"));
}
#endif

// ============================================================================
// 测试 2: Permute 步长魔法与转置
// ============================================================================
void check_permute(Device device) {
    auto opt = miniDL::device(device).requiresGrad(true);
    // [2, 3, 4] 的 3D 张量
    Tensor x = Tensor::empty(Shape({2, 3, 4}), opt);

    // 我们把维度换成 [4, 2, 3]，对应的索引是 {2, 0, 1}
    Tensor y = x.permute({2, 0, 1});

    // 1. 验证新形状
    EXPECT_EQ(y.shape()[0], 4);
    EXPECT_EQ(y.shape()[1], 2);
    EXPECT_EQ(y.shape()[2], 3);

    // 2. 【硬核验证】：验证底层的 Stride 是否正确排列！
    // X 原本的 strides 应该是: [3*4, 4, 1] = [12, 4, 1]
    // Y 的 strides 应该跟着维度换位变成: [1, 12, 4]
    EXPECT_EQ(y.impl()->strides()[0], 1);
    EXPECT_EQ(y.impl()->strides()[1], 12);
    EXPECT_EQ(y.impl()->strides()[2], 4);

    // 3. 验证零拷贝
    EXPECT_EQ(x.data_ptr<float>(), y.data_ptr<float>());

    // 4. 反向传播连通性验证
    Tensor grad_out = Tensor::ones(Shape({4, 2, 3}), miniDL::device(device));
    y.impl()->set_grad(grad_out.shared_impl());
    y.backward();

    // X 的梯度形状必须恢复成 [2, 3, 4]
    EXPECT_TRUE(x.grad().defined());
    EXPECT_EQ(x.grad().shape()[0], 2);
    EXPECT_EQ(x.grad().shape()[1], 3);
    EXPECT_EQ(x.grad().shape()[2], 4);
}

TEST_F(ShapeTest, PermuteCPU) {
    check_permute(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(ShapeTest, PermuteCUDA) {
    check_permute(Device("cuda:0"));
}
#endif

// ============================================================================
// 测试 3: Contiguous 连续化内存重排验证
// ============================================================================
void check_contiguous(Device device) {
    auto opt = miniDL::device(device).requiresGrad(true);

    // 1. 造一个连续的 [2, 3] 张量
    Tensor x_cpu = Tensor::empty(Shape({2, 3}), miniDL::device("cpu"));
    float* x_ptr = x_cpu.data_ptr<float>();
    // 填充物理内存: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
    for (int i = 0; i < 6; ++i) { x_ptr[i] = static_cast<float>(i); }

    Tensor x = x_cpu.to(device);
    x.impl()->set_requires_grad(true);

    // 2. 施加零拷贝魔法，使其不连续
    // 原 strides: [3, 1]  -> 新 strides: [1, 3]
    Tensor y = x.permute({1, 0});

    // 验证 y 是不连续的
    EXPECT_FALSE(y.impl()->is_contiguous());

    // 3. 呼叫物理拷贝魔法：强制连续化
    Tensor z = y.contiguous();

    // 验证 z 是连续的
    EXPECT_TRUE(z.impl()->is_contiguous());

    // 【硬核验证 1：物理深拷贝检查】
    // z 必须申请了新内存，指针绝对不能和 y (或 x) 一样！
    EXPECT_NE(y.data_ptr<float>(), z.data_ptr<float>());

    // 【硬核验证 2：物理内存数据透视】
    // 逻辑上的 y 应该是:
    // [0, 3]
    // [1, 4]
    // [2, 5]
    // 所以 z 的物理内存必须被重新排列成: 0, 3, 1, 4, 2, 5
    Tensor z_cpu = z.to(Device("cpu"));
    float* z_ptr = z_cpu.data_ptr<float>();
    EXPECT_FLOAT_EQ(z_ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(z_ptr[1], 3.0f);
    EXPECT_FLOAT_EQ(z_ptr[2], 1.0f);
    EXPECT_FLOAT_EQ(z_ptr[3], 4.0f);
    EXPECT_FLOAT_EQ(z_ptr[4], 2.0f);
    EXPECT_FLOAT_EQ(z_ptr[5], 5.0f);

    // 4. 反向传播连通性验证
    Tensor grad_out = Tensor::ones(Shape({3, 2}), miniDL::device(device));
    z.impl()->set_grad(grad_out.shared_impl());
    z.backward();

    // 验证经过 Contiguous -> Permute 的双重反向传播，梯度是否安全回到了起点 x
    EXPECT_TRUE(x.grad().defined());
    EXPECT_EQ(x.grad().shape()[0], 2);
    EXPECT_EQ(x.grad().shape()[1], 3);
}

TEST_F(ShapeTest, ContiguousCPU) {
    check_contiguous(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(ShapeTest, ContiguousCUDA) {
    check_contiguous(Device("cuda:0"));
}
#endif