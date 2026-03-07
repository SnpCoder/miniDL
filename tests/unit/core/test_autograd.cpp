#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/ops/math/add.h"
#include "../../../include/ops/math/mul.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

class AutogradEngineTest : public ::testing::Test {
   protected:
    void SetUp() override {}
};

// 测试 1: Autograd 动态计算图与梯度累加终极测试
TEST_F(AutogradEngineTest, AddAutogradGraph) {
    // 方程: y = 2 * x1 + x2 + 3.0
    auto opt  = miniDL::device("cpu").requiresGrad(true);
    Tensor x1 = Tensor::ones(Shape({1}), opt);
    Tensor x2 = Tensor::ones(Shape({1}), opt);

    Tensor t1 = x1 + x2;
    Tensor t2 = x1 + 3.0f;
    Tensor y  = t1 + t2;

    y.backward();

    ASSERT_TRUE(x1.grad().defined());
    ASSERT_TRUE(x2.grad().defined());

    EXPECT_FLOAT_EQ(x1.grad().data_ptr<float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(x2.grad().data_ptr<float>()[0], 1.0f);
}

TEST_F(AutogradEngineTest, MulAutogradGraph) {
    // 构造方程: f(x) = 3 * x * x + 2 * x
    // 这是个经典的抛物线，导数 f'(x) = 6 * x + 2
    // 当 x = 2.0 时，期待梯度：f'(2) = 14.0

    auto opt               = miniDL::device("cpu").requiresGrad(true);
    Tensor x               = Tensor::ones(Shape({1}), opt);
    x.data_ptr<float>()[0] = 2.0f;  // x 初始化为 2.0

    // 构建动态图
    Tensor t1 = x * x;      // x^2
    Tensor t2 = t1 * 3.0f;  // 3x^2
    Tensor t3 = x * 2.0f;   // 2x
    Tensor y  = t2 + t3;    // 3x^2 + 2x

    y.backward();

    // 严苛验证
    MINIDL_PRINT("Gradient of x (expected 14.0):");
    x.grad().print();
    EXPECT_FLOAT_EQ(x.grad().data_ptr<float>()[0], 14.0f);
}

#ifdef USE_CUDA
// 测试 2: GPU 上的 Autograd 引擎测试与大内存极限测试
TEST_F(AutogradEngineTest, CUDAAutogradGraph) {
    auto opt = miniDL::device("cuda:0").requiresGrad(true);

    Tensor x = Tensor::zeros(Shape({1024, 1024}), opt);

    // y = x + x + x  => dy/dx = 3.0
    Tensor y = x + x;
    y        = y + x;

    y.backward();

    Tensor grad_cpu = x.grad().to(Device("cpu"));
    EXPECT_FLOAT_EQ(grad_cpu.data_ptr<float>()[0], 3.0f);
    EXPECT_FLOAT_EQ(grad_cpu.data_ptr<float>()[1048575], 3.0f);
}
#endif

#ifdef USE_CUDA
TEST_F(AutogradEngineTest, CUDANonLinearAutogradGraph) {
    MINIDL_PRINT("========== 启动 CUDA 非线性计算图 Autograd 测试 ==========");

    // 构造方程: f(x) = 3 * x^2 + 2 * x
    // 导数 f'(x) = 6 * x + 2
    // 当 x = 2.0 时，期待梯度：f'(2) = 14.0

    auto opt = miniDL::device("cuda:0").requiresGrad(true);

    // 使用 100 万个元素的超大张量，压榨 float4 的极限带宽，并考验显存是否泄露
    Shape large_shape({1024, 1024});
    Tensor x = Tensor::ones(large_shape, opt);

    // x 原本是 1.0，我们利用刚写好的原地标量加法，把它全变成 2.0
    x += 1.0f;

    // 构建复杂的非线性动态图 (全链路均在 GPU 显存内飙车)
    Tensor t1 = x * x;      // x^2
    Tensor t2 = t1 * 3.0f;  // 3x^2
    Tensor t3 = x * 2.0f;   // 2x
    Tensor y  = t2 + t3;    // y = 3x^2 + 2x

    // 呼叫 GPU Autograd 引擎！
    y.backward();

    // 严苛验证
    ASSERT_TRUE(x.grad().defined());

    // 将计算好的庞大梯度搬回 CPU 抽样验毒
    Tensor grad_cpu = x.grad().to(Device("cpu"));

    MINIDL_PRINT("抽样检查 GPU 算出的梯度 (期待 14.0): {}", grad_cpu.data_ptr<float>()[0]);

    // 检查头部、中部和尾部（确保 float4 没越界）
    EXPECT_FLOAT_EQ(grad_cpu.data_ptr<float>()[0], 14.0f);
    EXPECT_FLOAT_EQ(grad_cpu.data_ptr<float>()[500000], 14.0f);
    EXPECT_FLOAT_EQ(grad_cpu.data_ptr<float>()[1048575], 14.0f);

    MINIDL_PRINT("========== CUDA 非线性 Autograd 测试完美通过 ==========");
}
#endif