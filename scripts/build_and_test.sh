#!/bin/bash

# ==============================================================================
# miniDL 一键构建与测试脚本
# ==============================================================================

# 【防呆设计 1】：遇到任何一条命令报错（比如编译失败），脚本立刻停止，绝不往下盲目执行
set -e

echo "🚀 开始构建与测试 miniDL 框架..."

# 【防呆设计 2】：无论你在哪个目录下运行这个脚本，它都会自动帮你切回 miniDL 的根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 如果 build 目录不存在，自动为你创建
if [ ! -d "build" ]; then
    echo "📁 发现 build 目录不存在，正在自动创建..."
    mkdir build
fi

cd build

echo "⚙️ [1/3] 运行 CMake 配置环境..."
cmake -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_HOST_COMPILER=g++-12 ..

echo "🔨 [2/3] 开始并行编译..."
# 【性能优化】：使用 $(nproc) 自动检测你的电脑有几个逻辑核心，直接火力全开
make -j$(nproc)

echo "🧪 [3/3] 运行 GTest 单元测试..."
ctest --output-on-failure