# 定义测试统一的输出路径 (编译出的可执行文件会统一存放在 build/bin/unitTest 里)
set(UNIT_TEST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/unitTest)

# ==========================================
# 1. 搜集所有的测试源文件 (极简扩展法)
# 以后你再加 tensor/test_tensor.cpp，只需要加一行路径即可
# ==========================================
file(GLOB UNIT_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/*.cpp
)

# 2. 声明统一的测试可执行文件
add_executable(minidl_unit_tests ${UNIT_TEST_SRCS})

# 3. 核心魔法：Target-based 链接
target_link_libraries(minidl_unit_tests 
    PRIVATE 
    minidl 
    GTest::gtest_main
)

# 4. 设置输出目录属性
set_target_properties(minidl_unit_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 5. 自动发现并注册 GTest
include(GoogleTest)
gtest_discover_tests(minidl_unit_tests)