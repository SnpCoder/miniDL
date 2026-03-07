#include "../../include/memory/allocator.h"

#include <cstdlib>

#include "../../include/utils/exception.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
namespace miniDL {
#ifndef MEM_ALIGNMENT_SIZE
#define MEM_ALIGNMENT_SIZE 128
#endif

void* CPUAllocator::allocate(size_t nbytes) {
    void* ptr = std::aligned_alloc(MEM_ALIGNMENT_SIZE, nbytes);
    if (ptr == nullptr) { throw std::bad_alloc(); }
    return ptr;
}

void CPUAllocator::deallocate(void* ptr) {
    if (ptr) { free(ptr); }
}

void* GPUAllocator::allocate(size_t nbytes) {
#ifdef USE_CUDA
    cudaError_t err = cudaSetDevice(_device_id);
    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("Failed to set CUDA device {}: {}", _device_id,
                             cudaGetErrorString(err));
    }
    const size_t alignmentSize = (nbytes + MEM_ALIGNMENT_SIZE - 1) & ~(MEM_ALIGNMENT_SIZE - 1);
    void* ptr                  = nullptr;
    err                        = cudaMalloc(&ptr, alignmentSize);
    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("Failed to allocate CUDA memory on device {}: {}", _device_id,
                             cudaGetErrorString(err));
    }
    return ptr;
#else
    MINIDL_THROW_RUNTIME("CUDA support is not enabled");
#endif
}

void GPUAllocator::deallocate(void* ptr) {
#ifdef USE_CUDA
    cudaError_t err = cudaSetDevice(_device_id);

    if (err == cudaErrorCudartUnloading) {
        return;  // CUDA runtime is unloading, memory will be freed by OS, just return
    };

    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("Failed to set CUDA device {}: {}", _device_id,
                             cudaGetErrorString(err));
    }
    err = cudaFree(ptr);

    if (err == cudaErrorCudartUnloading) {
        return;  // CUDA runtime is unloading, memory will be freed by OS, just return
    }

    if (err != cudaSuccess) {
        MINIDL_THROW_RUNTIME("Failed to free CUDA memory on device {}: {}", _device_id,
                             cudaGetErrorString(err));
    }
#else
    MINIDL_THROW_RUNTIME("CUDA support is not enabled");
#endif
}

std::unique_ptr<Allocator> AllocatorFactory::createAllocator(Device d) {
    if (d.isCpu()) {
        return std::make_unique<CPUAllocator>();
    } else if (d.isCuda()) {
        return std::make_unique<GPUAllocator>(d.index());
    } else {
        MINIDL_THROW_INVALID_ARG("Unsupported device type");
    }
}
}  // namespace miniDL
