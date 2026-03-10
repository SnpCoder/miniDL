#include "../../include/ops/factory.h"

#include <algorithm>
#include <cstring>
#include <random>

#include "../../include/utils/exception.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>

#include "../../include/ops/cuda/fill_cuda.cuh"
#endif

namespace miniDL {
namespace ops {

void fill_zeros(Tensor& t) {
    void* ptr          = t.impl()->data();
    size_t total_bytes = t.element_num() * get_element_size(t.data_type());

    if (t.device().isCpu()) {
        std::memset(ptr, 0, total_bytes);
    } else if (t.device().isCuda()) {
#ifdef USE_CUDA
        cudaMemset(ptr, 0, total_bytes);
#else
        MINIDL_THROW_RUNTIME("Compiled without CUDA support");
#endif
    }
}

void fill_ones(Tensor& t) {
    float* ptr = t.data_ptr<float>();
    size_t n   = t.element_num();

    if (t.device().isCpu()) {
        std::fill(ptr, ptr + n, 1.0f);
    } else if (t.device().isCuda()) {
#ifdef USE_CUDA
        cuda::launch_fill_kernel_float32(ptr, 1.0f, n);
#endif
    }
}

void fill_uniform(Tensor& t, float low, float high) {
    size_t n = t.element_num();
    std::vector<float> cpu_data(n);  // 临时 CPU 内存

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < n; ++i) { cpu_data[i] = dist(gen); }

    if (t.device().isCpu()) {
        std::memcpy(t.data_ptr<float>(), cpu_data.data(), n * sizeof(float));
    } else if (t.device().isCuda()) {
#ifdef USE_CUDA
        cudaMemcpy(t.data_ptr<float>(), cpu_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
#endif
    }
}

void fill_randn(Tensor& t, float mean, float std) {
    size_t n = t.element_num();
    std::vector<float> cpu_data(n);  // 临时 CPU 内存

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(mean, std);
    for (size_t i = 0; i < n; ++i) { cpu_data[i] = dist(gen); }

    if (t.device().isCpu()) {
        std::memcpy(t.data_ptr<float>(), cpu_data.data(), n * sizeof(float));
    } else if (t.device().isCuda()) {
#ifdef USE_CUDA
        cudaMemcpy(t.data_ptr<float>(), cpu_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
#endif
    }
}

}  // namespace ops
}  // namespace miniDL