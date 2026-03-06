#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/ops/math/add.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

// ============================================================================
// Add 算子专属测试套件
// ============================================================================
class AddOperatorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // 每个测试前的初始化工作
        MINIDL_INFO("Setting up AddOperatorTest...");
    }
};

// 1. 纯粹的数学正确性测试 (不需要求导)
TEST_F(AddOperatorTest, ForwardMathCorrectness) {
    auto opt = miniDL::device("cpu").requiresGrad(false);
    Tensor a = Tensor::zeros(Shape({2, 3}), opt);
    Tensor b = Tensor::zeros(Shape({2, 3}), opt);

    // a = [1.0, 1.0, ...]
    for (size_t i = 0; i < a.element_num(); ++i) a.data_ptr<float>()[i] = 1.0f;
    // b = [2.0, 2.0, ...]
    for (size_t i = 0; i < b.element_num(); ++i) b.data_ptr<float>()[i] = 2.0f;

    Tensor c = a + b;

    EXPECT_EQ(c.shape(), Shape({2, 3}));
    EXPECT_FALSE(c.impl()->requires_grad());  // 输入都不需要求导，输出必定不需要

    // 验证 C = [3.0, 3.0, ...]
    const float* c_ptr = c.data_ptr<float>();
    for (size_t i = 0; i < c.element_num(); ++i) { EXPECT_FLOAT_EQ(c_ptr[i], 3.0f); }
}

// 2. 动态计算图连通性测试 (极其关键)
TEST_F(AddOperatorTest, AutogradGraphConstruction) {
    Tensor a = Tensor::zeros(Shape({2, 2}), miniDL::device("cpu").requiresGrad(true));
    Tensor b = Tensor::zeros(Shape({2, 2}), miniDL::device("cpu").requiresGrad(false));

    Tensor c = a + b;

    // 只要有一个输入 requires_grad=true，输出就必须被追踪！
    EXPECT_TRUE(c.impl()->requires_grad());

    // C 必须知道它是被 Add 算子创造出来的
    EXPECT_NE(c.impl()->creator(), nullptr);

    // 【边界测试】B 是常数，所以它不应该有创造者
    EXPECT_EQ(b.impl()->creator(), nullptr);
}

// 3. 异常与防呆测试
TEST_F(AddOperatorTest, ExceptionHandling) {
    Tensor a = Tensor::zeros(Shape({2, 2}));
    Tensor b = Tensor::zeros(Shape({3, 3}));  // 形状不同

    // 期望抛出无效参数异常 (在我们实现 Broadcast 之前)
    EXPECT_THROW(
        { Tensor c = a + b; }, std::invalid_argument);  // 替换为你自定义异常的基础 std 类型
}

// for gpu --------------------------------------------------
// ============================================================================
// CUDA 专属向量化与边界测试
// ============================================================================
#ifdef USE_CUDA

TEST_F(AddOperatorTest, CUDAVectorizedExactMatch) {
    // 测试 1：元素个数恰好是 4 的倍数，完全触发 float4 向量化
    // 形状 1024 x 1024 = 1,048,576 个元素
    auto opt = miniDL::device("cuda:0").requiresGrad(true);
    Tensor a = Tensor::zeros(Shape({1024, 1024}), opt);
    Tensor b = Tensor::zeros(Shape({1024, 1024}), opt);

    // 在 GPU 上执行加法：C = A + 2.0f
    Tensor c = a + 2.0f;
    // C + B = 2.0f + 0.0f
    Tensor d = c + b;

    EXPECT_EQ(d.device(), Device("cuda:0"));

    // 将结果拉回 CPU 进行断言验证
    Tensor d_cpu       = d.to(Device("cpu"));
    const float* d_ptr = d_cpu.data_ptr<float>();

    // 抽样检查首尾和中间，保证极速执行
    EXPECT_FLOAT_EQ(d_ptr[0], 2.0f);
    EXPECT_FLOAT_EQ(d_ptr[500000], 2.0f);
    EXPECT_FLOAT_EQ(d_ptr[1048575], 2.0f);
}

TEST_F(AddOperatorTest, CUDAUnalignedBoundary) {
    // 测试 2：元素个数不是 4 的倍数 (比如 1023)，强行触发 else 里的零头处理逻辑
    auto opt = miniDL::device("cuda:0").requiresGrad(false);
    Tensor a = Tensor::zeros(Shape({1023}), opt);
    Tensor b = Tensor::zeros(Shape({1023}), opt);

    // 把 a 搬回 cpu 赋初值，再搬回 gpu（模拟真实数据加载）
    Tensor a_cpu = a.to(Device("cpu"));
    for (size_t i = 0; i < 1023; ++i) a_cpu.data_ptr<float>()[i] = static_cast<float>(i);
    a = a_cpu.to(Device("cuda:0"));

    // GPU 标量加法
    Tensor c = a + 10.0f;

    // 拉回 CPU 验证
    Tensor c_cpu       = c.to(Device("cpu"));
    const float* c_ptr = c_cpu.data_ptr<float>();

    // 第一个元素: 0 + 10 = 10
    EXPECT_FLOAT_EQ(c_ptr[0], 10.0f);
    // 最后一个元素(零头): 1022 + 10 = 1032
    EXPECT_FLOAT_EQ(c_ptr[1022], 1032.0f);
}

TEST_F(AddOperatorTest, CUDACrossDeviceException) {
    // 测试 3：防呆测试，GPU 张量和 CPU 张量相加必须抛出异常
    Tensor a_cpu = Tensor::zeros(Shape({10}), miniDL::device("cpu"));
    Tensor b_gpu = Tensor::zeros(Shape({10}), miniDL::device("cuda:0"));

    // 捕获跨设备相加异常
    EXPECT_THROW(
        { Tensor c = a_cpu + b_gpu; }, std::invalid_argument);  // 根据你抛出的具体异常类型调整
}

TEST_F(AddOperatorTest, CUDABackwardGraphCheck) {
    // 测试 4：GPU 算子的计算图连接测试
    Tensor a = Tensor::zeros(Shape({5}), miniDL::device("cuda:0").requiresGrad(true));
    Tensor b = Tensor::zeros(Shape({5}), miniDL::device("cuda:0").requiresGrad(false));

    Tensor c = a + b;
    Tensor d = c + 5.0f;

    // 确保 D 和 C 都在 GPU 上，并且都在计算图中
    EXPECT_TRUE(d.device().isCuda());
    EXPECT_TRUE(d.impl()->requires_grad());
    EXPECT_NE(d.impl()->creator(), nullptr);
}

#endif  // USE_CUDA