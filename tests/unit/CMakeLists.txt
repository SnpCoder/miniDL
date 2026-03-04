# 定义测试统一的输出路径 (编译出的可执行文件会统一存放在 build/bin/unitTest 里)
set(UNIT_TEST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/unitTest)

# 1. 声明你要编译的测试可执行文件
add_executable(test_shape utils/test_shape.cpp)

# 2. 核心魔法：Target-based 链接
# GTest::gtest_main 会自动帮你搞定 include 目录、静态库链接和 pthread！
target_link_libraries(test_shape 
    PRIVATE 
    minidl 
    GTest::gtest_main
)

# 3. 设置输出目录属性
set_target_properties(test_shape PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIT_TEST_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIT_TEST_OUTPUT_DIR}
)

# 4. 自动发现并注册 GTest
include(GoogleTest)
gtest_discover_tests(test_shape)