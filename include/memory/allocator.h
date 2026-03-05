#ifndef __MINIDL_ALLOCATOR_H__
#define __MINIDL_ALLOCATOR_H__

#include <cstddef>
#include <memory>

#include "../basicTypes.h"

namespace miniDL {

class Allocator {
   public:
    virtual ~Allocator() = default;

    virtual void* allocate(size_t nbytes) = 0;

    virtual void deallocate(void* ptr) = 0;

    virtual DeviceType getDeviceType() const = 0;
};

class CPUAllocator : public Allocator {
   public:
    void* allocate(size_t nbytes) override;

    void deallocate(void* ptr) override;

    DeviceType getDeviceType() const override { return DeviceType::kCPU; }
};

class GPUAllocator : public Allocator {
   private:
    int _device_id;

   public:
    explicit GPUAllocator(int deviceId = 0) : _device_id(deviceId) {}

    void* allocate(size_t nbytes) override;

    void deallocate(void* ptr) override;

    DeviceType getDeviceType() const override { return DeviceType::kGPU; }
};

class AllocatorFactory {
   public:
    static std::unique_ptr<Allocator> createAllocator(Device d);
};
}  // namespace miniDL
#endif  // __MINIDL_ALLOCATOR_H__