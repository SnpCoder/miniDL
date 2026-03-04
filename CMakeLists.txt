cmake_minimum_required(VERSION 3.14)
project(miniDL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_library(minidl INTERFACE)

target_include_directories(minidl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(minidl INTERFACE spdlog::spdlog)

enable_testing()
add_subdirectory(tests)