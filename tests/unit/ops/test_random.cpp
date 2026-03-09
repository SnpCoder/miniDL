#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

class RandomTest : public ::testing::Test {};

TEST_F(RandomTest, UniformDistributionCPU) {
    auto opt = miniDL::device("cpu").requiresGrad(true);
    Tensor t = Tensor::uniform(Shape({100, 100}), -0.5f, 0.5f, opt);

    EXPECT_EQ(t.shape(), Shape({100, 100}));
    EXPECT_TRUE(t.requires_grad());

    // 抽样验证边界
    float* ptr         = t.data_ptr<float>();
    bool within_bounds = true;
    for (int i = 0; i < 1000; ++i) {  // 抽查前1000个
        if (ptr[i] < -0.5f || ptr[i] > 0.5f) within_bounds = false;
    }
    EXPECT_TRUE(within_bounds);
}

#ifdef USE_CUDA
TEST_F(RandomTest, RandnDistributionCUDA) {
    auto opt = miniDL::device("cuda:0").requiresGrad(true);
    Tensor t = Tensor::randn(Shape({1024}), 0.0f, 1.0f, opt);

    EXPECT_TRUE(t.device().isCuda());
    EXPECT_TRUE(t.requires_grad());

    // 搬回 CPU 检查是否确实生成了非全零的数据
    Tensor cpu_t = t.to(Device("cpu"));
    EXPECT_NE(cpu_t.data_ptr<float>()[0], cpu_t.data_ptr<float>()[1]);  // 极大概率不相等
}
#endif