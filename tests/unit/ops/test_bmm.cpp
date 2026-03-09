#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

class BmmTest : public ::testing::Test {};

// ============================================================================
// 测试 1: BMM 严苛的数学前向与反向梯度验证 (小矩阵)
// ============================================================================
void check_bmm_math(Device device) {
    auto opt = miniDL::device(device).requiresGrad(true);

    // 1. 构造 A 矩阵: 形状 [2, 2, 3] (2个Batch, 每个是 2x3)
    Tensor a_cpu = Tensor::empty(Shape({2, 2, 3}), miniDL::device("cpu"));
    float* a_ptr = a_cpu.data_ptr<float>();
    // Batch 0
    a_ptr[0] = 1;
    a_ptr[1] = 2;
    a_ptr[2] = 3;
    a_ptr[3] = 4;
    a_ptr[4] = 5;
    a_ptr[5] = 6;
    // Batch 1
    a_ptr[6]  = 1;
    a_ptr[7]  = 1;
    a_ptr[8]  = 1;
    a_ptr[9]  = 1;
    a_ptr[10] = 1;
    a_ptr[11] = 1;

    // 2. 构造 B 矩阵: 形状 [2, 3, 2] (2个Batch, 每个是 3x2)
    Tensor b_cpu = Tensor::empty(Shape({2, 3, 2}), miniDL::device("cpu"));
    float* b_ptr = b_cpu.data_ptr<float>();
    // Batch 0
    b_ptr[0] = 1;
    b_ptr[1] = 2;
    b_ptr[2] = 3;
    b_ptr[3] = 4;
    b_ptr[4] = 5;
    b_ptr[5] = 6;
    // Batch 1
    b_ptr[6]  = 2;
    b_ptr[7]  = 2;
    b_ptr[8]  = 2;
    b_ptr[9]  = 2;
    b_ptr[10] = 2;
    b_ptr[11] = 2;

    Tensor A = a_cpu.to(device);
    Tensor B = b_cpu.to(device);
    A.impl()->set_requires_grad(true);
    B.impl()->set_requires_grad(true);

    // 3. 执行前向 BMM
    Tensor C = A.bmm(B);

    // 4. 前向数学验证
    EXPECT_EQ(C.shape()[0], 2);
    EXPECT_EQ(C.shape()[1], 2);
    EXPECT_EQ(C.shape()[2], 2);

    Tensor c_cpu = C.to(Device("cpu"));
    float* c_out = c_cpu.data_ptr<float>();
    // Batch 0: [[1*1+2*3+3*5, 1*2+2*4+3*6], ...] = [[22, 28], [49, 64]]
    EXPECT_FLOAT_EQ(c_out[0], 22.0f);
    EXPECT_FLOAT_EQ(c_out[1], 28.0f);
    EXPECT_FLOAT_EQ(c_out[2], 49.0f);
    EXPECT_FLOAT_EQ(c_out[3], 64.0f);
    // Batch 1: 全 1 矩阵 * 全 2 矩阵 = 全 6
    EXPECT_FLOAT_EQ(c_out[4], 6.0f);
    EXPECT_FLOAT_EQ(c_out[5], 6.0f);
    EXPECT_FLOAT_EQ(c_out[6], 6.0f);
    EXPECT_FLOAT_EQ(c_out[7], 6.0f);

    // 5. 反向传播验证 (核心护城河！)
    // 我们手动给 dC 注入全 1 的梯度
    Tensor grad_out = Tensor::ones(Shape({2, 2, 2}), miniDL::device(device));
    C.impl()->set_grad(grad_out.shared_impl());
    C.backward();

    // 6. 验证 dA = dC * B^T
    Tensor grad_a_cpu = A.grad().to(Device("cpu"));
    float* ga_ptr     = grad_a_cpu.data_ptr<float>();
    // dA Batch 0 的第一行应该是: [1,1] * [[1,3,5], [2,4,6]]^T (列相加) = [3, 7, 11]
    EXPECT_FLOAT_EQ(ga_ptr[0], 3.0f);
    EXPECT_FLOAT_EQ(ga_ptr[1], 7.0f);
    EXPECT_FLOAT_EQ(ga_ptr[2], 11.0f);

    // 7. 验证 dB = A^T * dC
    Tensor grad_b_cpu = B.grad().to(Device("cpu"));
    float* gb_ptr     = grad_b_cpu.data_ptr<float>();
    // dB Batch 0 的第一行应该是: [[1,4], [2,5], [3,6]]^T 的第一行 [1,4] * [1,1]^T = 5
    EXPECT_FLOAT_EQ(gb_ptr[0], 5.0f);
    EXPECT_FLOAT_EQ(gb_ptr[1], 5.0f);
}

TEST_F(BmmTest, MathCPU) {
    check_bmm_math(Device("cpu"));
}
#ifdef USE_CUDA
TEST_F(BmmTest, MathCUDA) {
    check_bmm_math(Device("cuda:0"));
}
#endif

// ============================================================================
// 测试 2: 触发 Vectorized float4 极速内核的大尺度测试
// ============================================================================
#ifdef USE_CUDA
void check_bmm_vectorized() {
    Device device("cuda:0");
    auto opt = miniDL::device(device).requiresGrad(false);

    // 强制构造 K 和 N 是 4 的倍数的大矩阵，以触发 V4 Float4 加速内核！
    // A: [8, 128, 64]   B: [8, 64, 256]   -> C: [8, 128, 256]
    Tensor A = Tensor::ones(Shape({8, 128, 64}), opt);
    Tensor B = Tensor::ones(Shape({8, 64, 256}), opt);

    // 矩阵乘法
    Tensor C = A.bmm(B);

    // 验证结果：两个全 1 的矩阵相乘，结果的每个元素必须严格等于 K (即 64.0)
    Tensor c_cpu = C.to(Device("cpu"));
    float* c_out = c_cpu.data_ptr<float>();

    // 随机抽查几个点，确保没有出现显存越界或地址对齐灾难
    EXPECT_FLOAT_EQ(c_out[0], 64.0f);
    EXPECT_FLOAT_EQ(c_out[1024], 64.0f);
    EXPECT_FLOAT_EQ(c_out[8 * 128 * 256 - 1], 64.0f);  // 最后一个元素
}

TEST_F(BmmTest, VectorizedCUDA) {
    check_bmm_vectorized();
}
#endif