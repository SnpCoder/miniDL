# 提交规范指南 (Commit Convention Guide)

本项目严格采用 [Conventional Commits](https://www.conventionalcommits.org/) (约定式提交) 规范。良好的提交习惯不仅有助于回溯代码历史，更是自动生成 Changelog 的基础。

## 1. 提交格式

每次代码提交的信息都必须遵循以下格式：

```text
<type>(<scope>): <subject>
```

**注意：**
- 冒号后面的空格是必须的！
- 尽量使用英文撰写提交信息，保持开源项目的通用性。

## 2. Type (类型) 列表
| **Type**     | **适用场景**                           | **示例**                                           |
| ------------ | -------------------------------------- | -------------------------------------------------- |
| **feat**     | 新增功能或算子                         | `feat(cuda): add leaky relu forward kernel`        |
| **fix**      | 修复 Bug 或内存泄漏                    | `fix(autograd): fix memory leak in backward graph` |
| **refactor** | 代码重构（无新功能，无 Bug 修复）      | `refactor(tensor): extract memory allocator logic` |
| **perf**     | 性能优化                               | `perf(cpu): use openmp to speedup matmul`          |
| **test**     | 新增或修改单元测试                     | `test(nn): add unit tests for linear layer`        |
| **docs**     | 仅更新文档或代码注释                   | `docs: update build instructions in README`        |
| **chore**    | 构建工具、依赖库更新等杂项             | `chore: update CMakeLists.txt to require C++17`    |
| **style**    | 格式化代码（空格、分号等，不影响逻辑） | `style: format src/backend using clang-format`     |

## 3. Scope (作用域) 参考##
Scope 用于指明本次提交影响的代码模块，虽然是可选的，但在大型框架中强烈建议填写。本项目推荐的 Scope 包含但不限于：
- `tensor`：张量数据结构与底层内存管理
- `autograd`：自动微分引擎与计算图
- `cpu`：CPU 算子与后端实现
- `cuda`：CUDA 算子与 GPU 管理
- `nn`：神经网络层与损失函数
- `optim`：优化器 (SGD, Adam 等)
- `build`：构建系统 (CMake, CI/CD)

## 4. 最佳与错误实践 ##
### ✅ 好的示例 (Good) ###
- `feat(autograd): implement topological sort for backward pass`
- `fix(tensor): resolve out-of-bounds memory access in reshape`
- `docs: add API reference for Module class`
### ❌ 错误的示例 (Bad) ###
- `update files` *(缺少类型和具体的描述)*
- `Fix bug` *(类型应该首字母小写，且缺少具体描述)*
- `feat(cuda):Added new kernel` *(冒号后缺少空格，描述不应大写开头且避免用过去式)*