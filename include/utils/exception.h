#ifndef __MINIDL_EXCEPTION_H__
#define __MINIDL_EXCEPTION_H__

#include <cstring>
#include <stdexcept>
#include <string>

// CUDA 核函数中不包含复杂的 C++ 库
#ifndef __CUDA_ARCH__
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#endif

namespace miniDL {

// 工具函数：获取单纯的文件名（编译期计算更佳，但运行时也足够快）
inline const char* basename(const char* filepath) {
    const char* last_slash     = strrchr(filepath, '/');
    const char* last_backslash = strrchr(filepath, '\\');
    const char* last_sep       = last_slash;
    if (last_backslash && (!last_slash || last_backslash > last_slash)) {
        last_sep = last_backslash;
    }
    return last_sep ? last_sep + 1 : filepath;
}

}  // namespace miniDL

// GPU
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define MINIDL_THROW_IMPL(exception_type, format_str, ...)                                         \
    do {                                                                                           \
        printf("\033[31m[GPU Error %s:%s:%d] " format_str "\033[0m\n", miniDL::basename(__FILE__), \
               __FUNCTION__, __LINE__, ##__VA_ARGS__);                                             \
        asm("trap;");                                                                              \
    } while (false)
#else
// CPU
#define MINIDL_THROW_IMPL(exception_type, format_str, ...)                                        \
    do {                                                                                          \
        std::string _pure_msg = fmt::format("[{}:{}:{}] " format_str, miniDL::basename(__FILE__), \
                                            __FUNCTION__, __LINE__, ##__VA_ARGS__);               \
        spdlog::critical(_pure_msg);                                                              \
        throw exception_type(_pure_msg);                                                          \
    } while (false)
#endif

#define MINIDL_THROW_INVALID_ARG(format_str, ...) \
    MINIDL_THROW_IMPL(std::invalid_argument, format_str, ##__VA_ARGS__)
#define MINIDL_THROW_RUNTIME(format_str, ...) \
    MINIDL_THROW_IMPL(std::runtime_error, format_str, ##__VA_ARGS__)
#define MINIDL_THROW_OUT_OF_RANGE(format_str, ...) \
    MINIDL_THROW_IMPL(std::out_of_range, format_str, ##__VA_ARGS__)

#endif  // __MINIDL_EXCEPTION_H__