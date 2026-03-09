#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/ops/shape/broadcast.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

class BroadcastTest : public ::testing::Test {};

// ============================================================================
// 核心数学验证函数
// ============================================================================
void check_broadcast_math(Device device) {
    // 【修复】：先在 CPU 上分配并写入数据，防止 SegFault！
    Tensor x_cpu = Tensor::empty(Shape({3}), miniDL::device("cpu"));
    float* x_ptr = x_cpu.data_ptr<float>();
    x_ptr[0]     = 1.0f;
    x_ptr[1]     = 2.0f;
    x_ptr[2]     = 3.0f;

    // 然后再安全地搬运到目标设备，并开启梯度追踪
    Tensor x = x_cpu.to(device);
    x.impl()->set_requires_grad(true);

    Tensor y = broadcast_to(x, Shape({2, 3}));

    Tensor y_cpu = y.to(Device("cpu"));
    float* y_ptr = y_cpu.data_ptr<float>();
    EXPECT_FLOAT_EQ(y_ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(y_ptr[2], 3.0f);
    EXPECT_FLOAT_EQ(y_ptr[3], 1.0f);
    EXPECT_FLOAT_EQ(y_ptr[5], 3.0f);

    Tensor grad_out = Tensor::ones(Shape({2, 3}), miniDL::device(device));
    grad_out += 1.0f;  // 变成2.0

    y.impl()->set_grad(grad_out.shared_impl());
    y.backward();

    Tensor grad_x_cpu = x.grad().to(Device("cpu"));
    float* gx_ptr     = grad_x_cpu.data_ptr<float>();
    EXPECT_FLOAT_EQ(gx_ptr[0], 4.0f);
    EXPECT_FLOAT_EQ(gx_ptr[1], 4.0f);
    EXPECT_FLOAT_EQ(gx_ptr[2], 4.0f);
}

TEST_F(BroadcastTest, CPUForwardBackward) {
    check_broadcast_math(Device("cpu"));
}

#ifdef USE_CUDA
TEST_F(BroadcastTest, CUDAForwardBackward) {
    check_broadcast_math(Device("cuda:0"));
}
#endif