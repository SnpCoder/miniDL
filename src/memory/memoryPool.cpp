#include "../../include/memory/memoryPool.h"

#include <iostream>
#include <stdexcept>

#include "../../include/utils/branchPrediction.h"
#include "../../include/utils/exception.h"

namespace miniDL {
MemoryPool::MemoryPool(Device dev, size_t max_cache_size)
    : _dev(dev), _max_cache_size(max_cache_size), _total_allocated_from_allocator(0) {
    _allocator = AllocatorFactory::createAllocator(dev);
}

MemoryPool::~MemoryPool() {
    std::lock_guard<std::mutex> locker(mtx);
    for (auto& pair : _idleBlocksPool) { _allocator->deallocate(pair.second); }
    _idleBlocksPool.clear();

    if (!_allocatedBlocks.empty()) {
        MINIDL_THROW_RUNTIME("MemoryPool: Memory Leak detected! Unreleased blocks count: {}",
                             _allocatedBlocks.size());
    }
    _allocatedBlocks.clear();
}

size_t MemoryPool::allocateBytes(size_t nbytes) const {
    if (nbytes == 0) { return 0; }
    size_t smallMemoryThreshold = 512 * 1024;  // 512kb
    size_t minMemoryAlignment   = 256;         // 256b
    if (nbytes <= smallMemoryThreshold) {
        size_t allocateByte = std::max(nbytes, minMemoryAlignment);
        allocateByte -= 1;
        allocateByte |= allocateByte >> 1;
        allocateByte |= allocateByte >> 2;
        allocateByte |= allocateByte >> 4;
        allocateByte |= allocateByte >> 8;
        allocateByte |= allocateByte >> 16;
        allocateByte |= allocateByte >> 32;
        return allocateByte + 1;
    } else {
        size_t largeMemoryAlignment = 1024 * 1024;  // 1mb
        return (nbytes + largeMemoryAlignment - 1) & ~(largeMemoryAlignment - 1);
    }
}

void MemoryPool::trimCache() {
    while (_total_allocated_from_allocator > _max_cache_size && !_idleBlocksPool.empty()) {
        // default: clear the idle largest block
        auto it          = std::prev(_idleBlocksPool.end());
        void* ptr        = it->second;
        size_t blockSize = it->first;

        _idleBlocksPool.erase(it);
        _total_allocated_from_allocator -= blockSize;

        // when turn to CUDA/OS
        mtx.unlock();
        _allocator->deallocate(ptr);
        mtx.lock();
    }
}

void* MemoryPool::malloc(size_t nbytes) {
    if (nbytes == 0) { return nullptr; }
    size_t allocateByte = allocateBytes(nbytes);
    std::unique_lock<std::mutex> locker(mtx);

    // find suitable block in cache
    auto it = _idleBlocksPool.lower_bound(allocateByte);
    // there is suitable block in cache
    if (it != _idleBlocksPool.end()) {
        void* ptr = it->second;
        _idleBlocksPool.erase(it);
        _allocatedBlocks[ptr] = allocateByte;
        return ptr;
    }
    // cann't find, should malloc
    locker.unlock();
    void* ptr = _allocator->allocate(allocateByte);
    if (unlikely(ptr == nullptr)) {
        MINIDL_THROW_RUNTIME("MemoryPool: backend allocation failed for size {}", allocateByte);
    }
    locker.lock();

    _total_allocated_from_allocator += allocateByte;
    _allocatedBlocks[ptr] = allocateByte;

    trimCache();
    return ptr;
}

void MemoryPool::free(void* ptr) {
    if (ptr == nullptr) return;
    std::lock_guard<std::mutex> locker(mtx);

    auto it = _allocatedBlocks.find(ptr);
    if (unlikely(it == _allocatedBlocks.end())) {
        MINIDL_THROW_RUNTIME("MemoryPool: freeing unallocated pointer or double free: {}", ptr);
        return;
    }

    size_t blockSize = it->second;
    _allocatedBlocks.erase(it);
    _idleBlocksPool.insert({blockSize, ptr});
}

MemoryPoolManager& MemoryPoolManager::getInstant() {
    static MemoryPoolManager ins;
    return ins;
}

MemoryPool* MemoryPoolManager::getMemoryPool(Device dev) {
    if (dev.isCpu()) {
        std::call_once(cpuInitFlag, [this, dev]() { cpuPool = std::make_unique<MemoryPool>(dev); });
        return cpuPool.get();
    } else if (dev.isCuda()) {
        // TODO: support multiple GPU devices
        std::call_once(gpuInitFlag, [this, dev]() { gpuPool = std::make_unique<MemoryPool>(dev); });
        return gpuPool.get();
    } else {
        MINIDL_THROW_INVALID_ARG("Unsupported device type");
        return nullptr;
    }
}
}  // namespace miniDL