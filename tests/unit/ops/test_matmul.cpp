#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/ops/math/matmul.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

class MatmulTest : public ::testing::Test {};

// ============================================================================
// 统一测试引擎：专为验证不同维度的 Matmul 前向与反向精度而设计
// 【严格遵守 M, N, K 传参顺序约定】
// ============================================================================
void check_matmul_correctness(int M, int N, int K, const std::string& strategy_name) {
    MINIDL_PRINT("========== 正在测试 CUDA 策略: {} [{}x{} * {}x{}] ==========", strategy_name, M,
                 K, K, N);

    auto opt = miniDL::device("cuda:0").requiresGrad(true);

    // A [M, K] 初始化为 2.0
    Tensor a = Tensor::ones(Shape({static_cast<size_t>(M), static_cast<size_t>(K)}), opt);
    a += 1.0f;

    // B [K, N] 初始化为 3.0
    Tensor b = Tensor::ones(Shape({static_cast<size_t>(K), static_cast<size_t>(N)}), opt);
    b += 2.0f;

    // 前向计算：Y = A * B
    Tensor y = mm(a, b);

    // 呼叫 GPU Autograd 大脑！
    y.backward();

    // 将张量搬回 CPU 进行极其严苛的断言验证
    Tensor y_cpu      = y.to(Device("cpu"));
    Tensor grad_a_cpu = a.grad().to(Device("cpu"));
    Tensor grad_b_cpu = b.grad().to(Device("cpu"));

    // 1. 验证前向传播 (Y = A * B)
    float expected_y = 6.0f * K;
    EXPECT_FLOAT_EQ(y_cpu.data_ptr<float>()[0], expected_y);
    EXPECT_FLOAT_EQ(y_cpu.data_ptr<float>()[y.element_num() - 1], expected_y);  // 检查尾部物理越界

    // 2. 验证 A 的梯度 (dy/dA = grad_out * B^T)
    float expected_grad_a = 3.0f * N;
    EXPECT_FLOAT_EQ(grad_a_cpu.data_ptr<float>()[0], expected_grad_a);
    EXPECT_FLOAT_EQ(grad_a_cpu.data_ptr<float>()[a.element_num() - 1], expected_grad_a);

    // 3. 验证 B 的梯度 (dy/dB = A^T * grad_out)
    float expected_grad_b = 2.0f * M;
    EXPECT_FLOAT_EQ(grad_b_cpu.data_ptr<float>()[0], expected_grad_b);
    EXPECT_FLOAT_EQ(grad_b_cpu.data_ptr<float>()[b.element_num() - 1], expected_grad_b);

    MINIDL_PRINT(">> [通过] {} 精度验证完美无瑕！", strategy_name);
}

// ============================================================================
// CUDA 智能分发器 (Auto-Dispatcher) 全面覆盖测试
// ============================================================================
#ifdef USE_CUDA

TEST_F(MatmulTest, Dispatcher_V0_Naive) {
    // 触发条件：最大的维度 <= 32
    // 预期：秒切 V0 朴素版，省去 Shared Memory 同步开销
    // 原调用(M,K,N): 16, 20, 32 -> 现调用(M,N,K): 16, 32, 20
    check_matmul_correctness(16, 32, 20, "V0 朴素实现 (Naive)");
}

TEST_F(MatmulTest, Dispatcher_V4_Vectorized) {
    // 触发条件：最大的维度 > 32，且 K 和 N 都是 4 的倍数
    // 预期：秒切 V4 性能怪兽，float4 极速显存榨汁
    // 原调用(M,K,N): 128, 256, 128 -> 现调用(M,N,K): 128, 128, 256
    check_matmul_correctness(128, 128, 256, "V4 向量化 (Vectorized float4)");
}

TEST_F(MatmulTest, Dispatcher_V1_SharedMemory) {
    // 触发条件：最大的维度 > 32，但 K 或 N 不是 4 的倍数 (比如 127/255)
    // 预期：平滑降级到 V1，防止 float4 越界导致内存段错误
    // 原调用(M,K,N): 127, 255, 127 -> 现调用(M,N,K): 127, 127, 255
    check_matmul_correctness(127, 127, 255, "V1 共享内存防越界 (Shared Memory)");
}

TEST_F(MatmulTest, Dispatcher_Extreme_Boundary) {
    // 触发条件：极端规模 1025，考验所有 Tiled 滑窗逻辑的边界处理 (零头块)
    // 注意：1025 > 32 且不是 4 的倍数，将由 V1 承担此次极限压测！
    check_matmul_correctness(1025, 1025, 1025, "极限边界压测 (1025x1025)");
}

#endif  // USE_CUDA

// ============================================================================
// CPU 对照与异常兜底测试
// ============================================================================
TEST_F(MatmulTest, CPUFallbackTest) {
    auto opt = miniDL::device("cpu").requiresGrad(true);
    // [2, 3] x [3, 2] -> [2, 2]
    Tensor a = Tensor::ones(Shape({2, 3}), opt);
    Tensor b = Tensor::ones(Shape({3, 2}), opt);

    a += 1.0f;  // a = 2.0
    b += 2.0f;  // b = 3.0

    Tensor y = mm(a, b);
    y.backward();

    // y = 6 * 3 = 18.0
    EXPECT_FLOAT_EQ(y.data_ptr<float>()[0], 18.0f);
    // grad_a = 3 * 2 = 6.0
    EXPECT_FLOAT_EQ(a.grad().data_ptr<float>()[0], 6.0f);
    // grad_b = 2 * 2 = 4.0
    EXPECT_FLOAT_EQ(b.grad().data_ptr<float>()[0], 4.0f);
}