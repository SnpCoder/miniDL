#include "../../include/memory/storage.h"

#include <cstring>

#include "../../include/memory/memoryPool.h"
#include "../../include/utils/branchPrediction.h"
#include "../../include/utils/exception.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace miniDL {
Storage::Storage(size_t size, Device dev) : _ptr(nullptr), _size(size), _dev(dev) {
    _pool = MemoryPoolManager::getInstant().getMemoryPool(dev);
    _ptr  = _pool->malloc(size);
}

Storage::~Storage() {
    if (_ptr != nullptr) {
        _pool->free(_ptr);
        _ptr = nullptr;
    }
}

Storage::Storage(Storage&& other) noexcept
    : _ptr(other._ptr), _size(other._size), _dev(other._dev), _pool(other._pool) {
    // avoid double free
    other._ptr  = nullptr;
    other._size = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        if (_ptr != nullptr) _pool->free(_ptr);
        _ptr  = other._ptr;
        _size = other._size;
        _dev  = other._dev;
        _pool = other._pool;
    }
    return *this;
}

std::shared_ptr<Storage> Storage::create(size_t size, Device dev) {
    return std::make_shared<Storage>(size, dev);
}

std::shared_ptr<Storage> Storage::toDevice(Device targetDev) {
    // copy to same device
    if (targetDev.type() == _dev.type() && targetDev.index() == _dev.index()) {
        // Same device, just return a copy
        auto new_storage = create(_size, _dev);
        if (_dev.isCpu()) {
            memcpy(new_storage->data(), _ptr, _size);
        } else {
#ifdef USE_CUDA
            // CUDA to CUDA copy
            cudaMemcpy(new_storage->data(), _ptr, _size, cudaMemcpyDeviceToDevice);
#else
            MINIDL_THROW_RUNTIME("Should compile with \"USE_CUDA\"");
#endif
        }
        return new_storage;
    }

    if (targetDev.isCpu()) {
        auto cpu_storage = create(_size, targetDev);
        if (_dev.isCuda()) {
            // GPU to CPU
#ifdef USE_CUDA
            cudaDeviceSynchronize();
            cudaError_t err = cudaMemcpy(cpu_storage->data(), _ptr, _size, cudaMemcpyDeviceToHost);
            if (unlikely(err != cudaSuccess)) {
                MINIDL_THROW_RUNTIME("CUDA memory copy failed: {}", cudaGetErrorString(err));
            }
#else
            MINIDL_THROW_RUNTIME("Should compile with \"USE_CUDA\"");
#endif
        } else {
            // CPU to CPU
            memcpy(cpu_storage->data(), _ptr, _size);
        }
        return cpu_storage;
    } else if (targetDev.isCuda()) {
        // Copy to CUDA
        auto cuda_storage = create(_size, targetDev);
        if (_dev.isCpu()) {
            // CPU to GPU
#ifdef USE_CUDA
            cudaMemcpy(cuda_storage->data(), _ptr, _size, cudaMemcpyHostToDevice);
#else
            MINIDL_THROW_RUNTIME("Should compile with \"USE_CUDA\"");
#endif
        } else {
            // GPU to GPU
#ifdef USE_CUDA
            cudaMemcpy(cuda_storage->data(), _ptr, _size, cudaMemcpyDeviceToDevice);
#else
            MINIDL_THROW_RUNTIME("Should compile with \"USE_CUDA\"");
#endif
        }
        return cuda_storage;
    } else {
        MINIDL_THROW_RUNTIME("Unsupported device type for transfer");
    }
}
}  // namespace miniDL