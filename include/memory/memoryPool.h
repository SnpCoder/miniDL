#ifndef __MINIDL_MEMORYPOOL_H__
#define __MINIDL_MEMORYPOOL_H__

#include <stddef.h>

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "allocator.h"

namespace miniDL {
class MemoryPool {
   private:
    mutable std::mutex mtx;

    std::unique_ptr<Allocator> _allocator;

    // blocks are idle
    std::multimap<size_t, void*> _idleBlocksPool;

    // blocks are used
    std::unordered_map<void*, size_t> _allocatedBlocks;

    size_t _total_allocated_from_allocator;

    size_t _max_cache_size;

    Device _dev;

    size_t allocateBytes(size_t nbytes) const;

    void trimCache();

   public:
    MemoryPool(Device d, size_t max_cache_size = 512 * 1024 * 1024);

    ~MemoryPool();

    void* malloc(size_t nbytes);

    void free(void* ptr);
};

class MemoryPoolManager {
   private:
    std::unique_ptr<MemoryPool> cpuPool;
    std::unique_ptr<MemoryPool> gpuPool;

    std::once_flag cpuInitFlag;
    std::once_flag gpuInitFlag;

    MemoryPoolManager()  = default;
    ~MemoryPoolManager() = default;

   public:
    MemoryPoolManager(const MemoryPoolManager&)            = delete;
    MemoryPoolManager& operator=(const MemoryPoolManager&) = delete;

    static MemoryPoolManager& getInstant();

    MemoryPool* getMemoryPool(Device d);
};
}  // namespace miniDL
#endif  // __MINIDL_MEMORYPOOL_H__