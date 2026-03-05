#include <gtest/gtest.h>

#include <vector>

#include "../../../include/memory/memoryPool.h"
#include "../../../include/memory/storage.h"

using namespace miniDL;

// ===================================================================
// 1. 基础分配测试 (CPU)
// ===================================================================
TEST(MemoryTest, CPUStorageBasic) {
    size_t bytes = 1024 * sizeof(float);
    // 极其优雅的 API 调用：直接传 Device("cpu")
    auto storage = Storage::create(bytes, Device("cpu"));

    EXPECT_EQ(storage->size(), bytes);
    
    // 提示：这要求你在 Storage 类里把原来的 deviceType() 和 deviceId() 
    // 合并成了一个直接返回 Device 对象的 getter： Device device() const { return _device; }
    // 并且我们之前在 Device 类里重载了 == 运算符，所以这里可以直接 EXPECT_EQ！
    EXPECT_EQ(storage->device(), Device("cpu")); 
    
    EXPECT_NE(storage->data(), nullptr);

    // 测试内存是否真正可写可读
    float* ptr = static_cast<float*>(storage->data());
    ptr[0]     = 3.14f;
    ptr[1023]  = 2.71f;

    EXPECT_EQ(ptr[0], 3.14f);
    EXPECT_EQ(ptr[1023], 2.71f);
}

// ===================================================================
// 2. 内存池核心机制验证：复用命中测试 (极其重要)
// ===================================================================
TEST(MemoryTest, MemoryPoolReuse) {
    void* first_ptr = nullptr;

    {
        // 作用域 1：申请 1MB 内存
        auto storage = Storage::create(1024 * 1024, Device("cpu"));
        first_ptr    = storage->data();
    }  // 离开作用域，storage 析构，内存被回收到 MemoryPool，并没有还给 OS

    {
        // 作用域 2：再次申请 1MB 内存
        auto storage2    = Storage::create(1024 * 1024, Device("cpu"));
        void* second_ptr = storage2->data();

        // 见证奇迹的时刻：由于缓存池的存在，底层指针必须完全一致！
        EXPECT_EQ(first_ptr, second_ptr) << "MemoryPool did not reuse the idle block!";
    }
}

// ===================================================================
// 3. GPU 与跨设备搬移测试 (仅在启用 CUDA 时编译)
// ===================================================================
#ifdef USE_CUDA
TEST(MemoryTest, GPUStorageAndTransfer) {
    size_t elements = 1000;
    size_t bytes    = elements * sizeof(float);

    // 1. 在 CPU 上创建数据并初始化
    auto cpu_storage = Storage::create(bytes, Device("cpu"));
    float* cpu_ptr   = static_cast<float*>(cpu_storage->data());
    for (size_t i = 0; i < elements; ++i) { cpu_ptr[i] = static_cast<float>(i); }

    // 2. CPU 拷贝到 GPU (极其直观的字符串 API: "cuda:0")
    auto gpu_storage = cpu_storage->toDevice(Device("cuda:0"));
    EXPECT_EQ(gpu_storage->device(), Device("cuda:0"));
    EXPECT_NE(gpu_storage->data(), nullptr);

    // 3. GPU 拷贝回 CPU
    auto cpu_storage_back = gpu_storage->toDevice(Device("cpu"));
    EXPECT_EQ(cpu_storage_back->device(), Device("cpu"));

    // 4. 验证数据完整性 (证明 cudaMemcpy 成功了)
    float* cpu_ptr_back = static_cast<float*>(cpu_storage_back->data());
    EXPECT_EQ(cpu_ptr_back[0], 0.0f);
    EXPECT_EQ(cpu_ptr_back[elements - 1], 999.0f);
    EXPECT_EQ(cpu_ptr_back[500], 500.0f);
}
#endif