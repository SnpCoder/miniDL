#include <gtest/gtest.h>

#include "../../../include/core/tensor.h"
#include "../../../include/core/tensorOptions.h"
#include "../../../include/utils/log.h"

using namespace miniDL;

// ============================================================================
// 1. TensorOptions 极致流式 API 测试
// ============================================================================
TEST(TensorTest, OptionsFluentAPI) {
    // 测试全局工厂函数链式调用
    auto opt = miniDL::device("cuda:0").dataType(DataType::kInt32).requiresGrad(true);

    EXPECT_TRUE(opt.device().isCuda());
    EXPECT_EQ(opt.device().index(), 0);
    EXPECT_EQ(opt.data_type(), DataType::kInt32);
    EXPECT_TRUE(opt.require_grad());

    // 测试隐式转换和默认值
    TensorOptions default_opt;
    EXPECT_TRUE(default_opt.device().isCpu());
    EXPECT_EQ(default_opt.data_type(), DataType::kFloat32);
}

// ============================================================================
// 2. Strides (步长) 与物理内存排布测试
// ============================================================================
TEST(TensorTest, ContiguousStridesCalculation) {
    // 1D 张量: shape {5} -> strides {1}
    Tensor t1 = Tensor::empty(Shape({5}));
    EXPECT_EQ(t1.impl()->strides().size(), 1);
    EXPECT_EQ(t1.impl()->strides()[0], 1);
    EXPECT_TRUE(t1.impl()->is_contiguous());

    // 2D 张量: shape {3, 4} -> strides {4, 1}
    Tensor t2 = Tensor::empty(Shape({3, 4}));
    EXPECT_EQ(t2.impl()->strides().size(), 2);
    EXPECT_EQ(t2.impl()->strides()[0], 4);
    EXPECT_EQ(t2.impl()->strides()[1], 1);
    EXPECT_TRUE(t2.impl()->is_contiguous());

    // 3D 张量: shape {2, 3, 4} -> strides {12, 4, 1}
    Tensor t3 = Tensor::empty(Shape({2, 3, 4}));
    EXPECT_EQ(t3.impl()->strides().size(), 3);
    EXPECT_EQ(t3.impl()->strides()[0], 12);
    EXPECT_EQ(t3.impl()->strides()[1], 4);
    EXPECT_EQ(t3.impl()->strides()[2], 1);
    EXPECT_TRUE(t3.impl()->is_contiguous());
}

// ============================================================================
// 3. 内存分配与 Zeros 初始化测试
// ============================================================================
TEST(TensorTest, CPUZerosInitialization) {
    Shape shape({2, 5});  // 10 个元素
    Tensor t = Tensor::zeros(shape, miniDL::dataType(DataType::kFloat32));

    EXPECT_TRUE(t.defined());
    EXPECT_EQ(t.element_num(), 10);
    EXPECT_EQ(t.device(), Device("cpu"));

    // 严格检查内存是否真正被清零
    float* ptr = t.data_ptr<float>();
    ASSERT_NE(ptr, nullptr);  // 确保指针不为空
    for (size_t i = 0; i < t.element_num(); ++i) { EXPECT_FLOAT_EQ(ptr[i], 0.0f); }

    // 写入数据测试可变性
    ptr[0] = 42.0f;
    EXPECT_FLOAT_EQ(ptr[0], 42.0f);
}

// ============================================================================
// 4. 浅拷贝与共享内存机制测试 (极其重要)
// ============================================================================
TEST(TensorTest, SharedMemorySemantics) {
    Tensor a = Tensor::zeros(Shape({10}));
    Tensor b = a;  // 触发拷贝构造

    // A 和 B 必须共享同一个 Impl 和 底层 Storage
    EXPECT_EQ(a.impl(), b.impl());
    EXPECT_EQ(a.data_ptr<float>(), b.data_ptr<float>());

    // 修改 B，A 也必须跟着变
    b.data_ptr<float>()[0] = 99.0f;
    EXPECT_FLOAT_EQ(a.data_ptr<float>()[0], 99.0f);
}

// ============================================================================
// 5. 跨设备硬拷贝测试 (CPU <-> GPU)
// ============================================================================
#ifdef USE_CUDA
TEST(TensorTest, CrossDeviceTransfer) {
    // 1. 在 CPU 准备非零数据
    Tensor cpu_t   = Tensor::empty(Shape({100}));
    float* cpu_ptr = cpu_t.data_ptr<float>();
    for (int i = 0; i < 100; ++i) { cpu_ptr[i] = static_cast<float>(i); }

    // 2. 搬到 GPU
    Tensor gpu_t = cpu_t.to(Device("cuda:0"));
    EXPECT_EQ(gpu_t.device(), Device("cuda:0"));

    // 由于是跨设备，底层指针绝不能相同！
    EXPECT_NE(gpu_t.impl(), cpu_t.impl());

    // 3. 在 GPU 上分配全 0 张量
    Tensor gpu_zeros = Tensor::zeros(Shape({100}), miniDL::device("cuda:0"));
    EXPECT_EQ(gpu_zeros.device(), Device("cuda:0"));

    // 4. 把 GPU 的张量搬回 CPU 进行断言验证
    Tensor back_to_cpu       = gpu_t.to(Device("cpu"));
    Tensor zeros_back_to_cpu = gpu_zeros.to(Device("cpu"));

    float* back_ptr       = back_to_cpu.data_ptr<float>();
    float* zeros_back_ptr = zeros_back_to_cpu.data_ptr<float>();

    // 验证数据完整性
    EXPECT_FLOAT_EQ(back_ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(back_ptr[99], 99.0f);
    EXPECT_FLOAT_EQ(zeros_back_ptr[50], 0.0f);
}

// ============================================================================
// 6. CPU print && GPU print 测试
// ============================================================================
TEST(TensorTest, PrintAndFormat) {
    // 构造一个 2x3 的 CPU 张量
    Tensor t = Tensor::zeros(Shape({2, 3}), miniDL::device("cpu").requiresGrad(true));

    // 用刚写的索引寻址功能，精准修改几个值
    t.data_ptr<float>()[t.impl()->compute_local_offset({0, 1})] = 3.14f;
    t.data_ptr<float>()[t.impl()->compute_local_offset({1, 2})] = 9.99f;

    MINIDL_PRINT("========== 视觉震撼：张量打印测试 ==========");
    t.print();
    MINIDL_PRINT("============================================\n");

#ifdef USE_CUDA
    // 测试极其硬核的 GPU 自动回退打印！
    Tensor gpu_t = t.to(Device("cuda:0"));
    MINIDL_PRINT("========== GPU 张量自动安全打印测试 ==========");
    gpu_t.print();
    MINIDL_PRINT("============================================\n");
#endif
}
#endif