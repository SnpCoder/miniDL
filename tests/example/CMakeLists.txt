# 1. 定义统一的输出路径 (编译出的可执行文件统一存放在 build/bin/examples 里)
set(EXAMPLE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/examples)

# 2. 声明独立的可执行文件
# 如果未来有更多的 example (比如 train_mlp.cpp)，可以在这里继续 add_executable
add_executable(train_linear train_linear.cpp)

# 3. 核心魔法：只链接 miniDL 核心库，绝对不链接 GTest！
target_link_libraries(train_linear 
    PRIVATE 
    minidl 
)

# 4. 设置输出目录属性，保持你原有的优雅结构
set_target_properties(train_linear PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${EXAMPLE_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXAMPLE_OUTPUT_DIR}
)