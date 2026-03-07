#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/core/tensorOptions.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

// ============================================================================
// 1. TensorOptions 极致流式 API 测试
// ============================================================================
TEST(TensorTest, OptionsFluentAPI) {
    auto opt = miniDL::device("cuda:0").dataType(DataType::kInt32).requiresGrad(true);

    EXPECT_TRUE(opt.device().isCuda());
    EXPECT_EQ(opt.device().index(), 0);
    EXPECT_EQ(opt.data_type(), DataType::kInt32);
    EXPECT_TRUE(opt.require_grad());

    TensorOptions default_opt;
    EXPECT_TRUE(default_opt.device().isCpu());
    EXPECT_EQ(default_opt.data_type(), DataType::kFloat32);
}

// ============================================================================
// 2. Strides (步长) 与物理内存排布测试
// ============================================================================
TEST(TensorTest, ContiguousStridesCalculation) {
    Tensor t1 = Tensor::empty(Shape({5}));
    EXPECT_EQ(t1.impl()->strides().size(), 1);
    EXPECT_EQ(t1.impl()->strides()[0], 1);
    EXPECT_TRUE(t1.impl()->is_contiguous());

    Tensor t2 = Tensor::empty(Shape({3, 4}));
    EXPECT_EQ(t2.impl()->strides().size(), 2);
    EXPECT_EQ(t2.impl()->strides()[0], 4);
    EXPECT_EQ(t2.impl()->strides()[1], 1);
    EXPECT_TRUE(t2.impl()->is_contiguous());

    Tensor t3 = Tensor::empty(Shape({2, 3, 4}));
    EXPECT_EQ(t3.impl()->strides().size(), 3);
    EXPECT_EQ(t3.impl()->strides()[0], 12);
    EXPECT_EQ(t3.impl()->strides()[1], 4);
    EXPECT_EQ(t3.impl()->strides()[2], 1);
    EXPECT_TRUE(t3.impl()->is_contiguous());
}

// ============================================================================
// 3. [升级版] 内存分配与 Zeros/Ones 极限填充测试
// ============================================================================
TEST(TensorTest, ZerosAndOnesInitialization) {
    // 采用 1023 这种奇葩数字，专门测试 float4 向量化的尾部越界保护逻辑
    Shape shape({1023});

    // CPU 测试
    Tensor cpu_zeros = Tensor::zeros(shape, miniDL::device("cpu"));
    Tensor cpu_ones  = Tensor::ones(shape, miniDL::device("cpu"));
    EXPECT_FLOAT_EQ(cpu_zeros.data_ptr<float>()[0], 0.0f);
    EXPECT_FLOAT_EQ(cpu_ones.data_ptr<float>()[1022], 1.0f);  // 检查尾部

#ifdef USE_CUDA
    // GPU 测试
    Tensor gpu_zeros = Tensor::zeros(shape, miniDL::device("cuda:0"));
    Tensor gpu_ones  = Tensor::ones(shape, miniDL::device("cuda:0"));

    // 拉回 CPU 验证
    Tensor check_zeros = gpu_zeros.to(Device("cpu"));
    Tensor check_ones  = gpu_ones.to(Device("cpu"));

    EXPECT_FLOAT_EQ(check_zeros.data_ptr<float>()[500], 0.0f);
    EXPECT_FLOAT_EQ(check_ones.data_ptr<float>()[1022], 1.0f);
#endif
}

// ============================================================================
// 4. 浅拷贝与共享内存机制测试 (极其重要)
// ============================================================================
TEST(TensorTest, SharedMemorySemantics) {
    Tensor a = Tensor::zeros(Shape({10}));
    Tensor b = a;  // 触发拷贝构造

    EXPECT_EQ(a.impl(), b.impl());
    EXPECT_EQ(a.data_ptr<float>(), b.data_ptr<float>());

    b.data_ptr<float>()[0] = 99.0f;
    EXPECT_FLOAT_EQ(a.data_ptr<float>()[0], 99.0f);
}

// ============================================================================
// 5. 跨设备硬拷贝测试 (CPU <-> GPU)
// ============================================================================
#ifdef USE_CUDA
TEST(TensorTest, CrossDeviceTransfer) {
    Tensor cpu_t   = Tensor::empty(Shape({100}));
    float* cpu_ptr = cpu_t.data_ptr<float>();
    for (int i = 0; i < 100; ++i) { cpu_ptr[i] = static_cast<float>(i); }

    Tensor gpu_t = cpu_t.to(Device("cuda:0"));
    EXPECT_EQ(gpu_t.device(), Device("cuda:0"));
    EXPECT_NE(gpu_t.impl(), cpu_t.impl());

    Tensor gpu_zeros = Tensor::zeros(Shape({100}), miniDL::device("cuda:0"));
    EXPECT_EQ(gpu_zeros.device(), Device("cuda:0"));

    Tensor back_to_cpu       = gpu_t.to(Device("cpu"));
    Tensor zeros_back_to_cpu = gpu_zeros.to(Device("cpu"));

    float* back_ptr       = back_to_cpu.data_ptr<float>();
    float* zeros_back_ptr = zeros_back_to_cpu.data_ptr<float>();

    EXPECT_FLOAT_EQ(back_ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(back_ptr[99], 99.0f);
    EXPECT_FLOAT_EQ(zeros_back_ptr[50], 0.0f);
}
#endif  // 修复：正确闭合跨设备测试的宏

// ============================================================================
// 6. CPU print && GPU print 测试
// ============================================================================
TEST(TensorTest, PrintAndFormat) {
    Tensor t = Tensor::zeros(Shape({2, 3}), miniDL::device("cpu").requiresGrad(true));

    t.data_ptr<float>()[t.impl()->compute_local_offset({0, 1})] = 3.14f;
    t.data_ptr<float>()[t.impl()->compute_local_offset({1, 2})] = 9.99f;

    MINIDL_PRINT("========== 视觉震撼：张量打印测试 ==========");
    t.print();
    MINIDL_PRINT("============================================\n");

#ifdef USE_CUDA  // 修复：只把 GPU 打印逻辑包起来
    Tensor gpu_t = t.to(Device("cuda:0"));
    MINIDL_PRINT("========== GPU 张量自动安全打印测试 ==========");
    gpu_t.print();
    MINIDL_PRINT("============================================\n");
#endif
}

// ============================================================================
// 7. [全新] 原地加法 (In-place Addition) 及其边界测试
// ============================================================================
TEST(TensorTest, InplaceAddition) {
    auto opt = miniDL::device("cpu");
    Tensor a = Tensor::ones(Shape({5}), opt);  // [1, 1, 1, 1, 1]
    Tensor b = Tensor::ones(Shape({5}), opt);  // [1, 1, 1, 1, 1]

    // 测试 += Tensor
    a += b;
    EXPECT_FLOAT_EQ(a.data_ptr<float>()[0], 2.0f);

    // 测试 += Scalar
    a += 3.0f;
    EXPECT_FLOAT_EQ(a.data_ptr<float>()[4], 5.0f);

#ifdef USE_CUDA
    // 测试 1025 这个特殊的长度，考验 float4 循环和尾部处理
    Tensor gpu_a = Tensor::ones(Shape({1025}), miniDL::device("cuda:0"));
    Tensor gpu_b = Tensor::ones(Shape({1025}), miniDL::device("cuda:0"));

    gpu_a += gpu_b;  // 全变 2.0
    gpu_a += 10.0f;  // 全变 12.0

    Tensor check_a = gpu_a.to(Device("cpu"));
    EXPECT_FLOAT_EQ(check_a.data_ptr<float>()[0], 12.0f);
    EXPECT_FLOAT_EQ(check_a.data_ptr<float>()[1024], 12.0f);  // 极其严苛的尾部越界检查
#endif
}