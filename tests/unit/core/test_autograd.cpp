#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/ops/math/add.h"
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