cmake_minimum_required(VERSION 3.17)
project(miniDL LANGUAGES CXX CUDA)

add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# 关闭 GTest 自带的测试和安装目标
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
# 关闭 Spdlog 自带的测试和示例程序
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)

set(GTEST_LIB
    ${PROJECT_BINARY_DIR}/lib/libgtest.a
    ${PROJECT_BINARY_DIR}/lib/libgtest_main.a
    pthread
)

set(GTEST_INCLUDE_DIR
    ${PROJECT_SOURCE_DIR}/thirdParty/googletest/include
    ${PROJECT_SOURCE_DIR}/thirdParty/googlemock/include
)
add_subdirectory(thirdParty/googletest)
add_subdirectory(thirdParty/spdlog)

file(
    GLOB_RECURSE MINIDL_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/memory/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/memory/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
)

add_library(minidl ${MINIDL_SRCS})

find_package(CUDAToolkit REQUIRED)

target_include_directories(minidl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(minidl PUBLIC USE_CUDA)
target_compile_options(minidl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=128>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=2417>
)

target_link_libraries(minidl PUBLIC spdlog::spdlog CUDA::cudart)

enable_testing()
add_subdirectory(tests)