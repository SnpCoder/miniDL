#!/bin/bash

# ==============================================================================
# miniDL Example 一键构建与运行脚本
# ==============================================================================

# 【防呆设计 1】：遇到任何一条命令报错，立刻停止
set -e

# 【防呆设计 2】：检查用户是否输入了要运行的 example 名称
if [ $# -lt 1 ]; then
    echo "❌ 错误：请指定要运行的 Example 名称！"
    echo "💡 用法: $0 <example_name> [args...]"
    echo "   例如: $0 train_linear"
    echo "   例如: $0 train_linear --cpu"
    echo "   例如: $0 train_linear -d 0"
    exit 1
fi

# 提取第一个参数作为可执行文件的名字
EXAMPLE_NAME=$1

# 【高阶技巧】：使用 shift 弹出第一个参数，剩下的所有参数存入 EXTRA_ARGS，
# 这样就可以把 '--cpu' 或 '-d 0' 完美透传给你的 C++ 程序！
shift
EXTRA_ARGS="$@"

echo "🚀 开始构建并运行 miniDL 示例: ${EXAMPLE_NAME}..."

# 自动切回根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d "build" ]; then
    echo "📁 发现 build 目录不存在，正在自动创建..."
    mkdir build
fi

cd build

echo "⚙️ [1/3] 运行 CMake 配置环境..."
# 保持与你单元测试相同的编译器环境
cmake -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_HOST_COMPILER=g++-12 ..

echo "🔨 [2/3] 开始并行编译..."
make -j$(nproc)

# 定义编译出的二进制文件路径
EXAMPLE_PATH="./bin/examples/${EXAMPLE_NAME}"

echo "🏃 [3/3] 启动执行: ${EXAMPLE_NAME} ${EXTRA_ARGS}"
echo "=============================================================================="

# 【防呆设计 3】：检查编译后的可执行文件到底存不存在
if [ -x "$EXAMPLE_PATH" ]; then
    # 真正执行 C++ 程序，并透传参数
    "$EXAMPLE_PATH" $EXTRA_ARGS
else
    echo "❌ 致命错误：找不到可执行文件 $EXAMPLE_PATH"
    echo "   可能的原因："
    echo "   1. 你输入的 example 名字拼写错误。"
    echo "   2. 你忘记在 tests/examples/CMakeLists.txt 中添加 add_executable(${EXAMPLE_NAME} ...) 了。"
    exit 1
fi

echo "=============================================================================="
echo "✅ 示例 ${EXAMPLE_NAME} 运行结束！"