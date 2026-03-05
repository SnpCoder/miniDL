#ifndef __ORIGIN_DL_BRANCH_PREDICTION_H__
#define __ORIGIN_DL_BRANCH_PREDICTION_H__

// ============================================================================
// 分支预测优化宏 (Branch Prediction Optimization Macros)
// ============================================================================

/**
 * @def likely(x)
 * @brief 告诉编译器条件 `x` 大概率为真 (True)。
 * @details 编译器会将 `if` 分支的汇编代码紧挨着当前代码块放置，以最大化指令缓存(I-Cache)命中率。
 */

/**
 * @def unlikely(x)
 * @brief 告诉编译器条件 `x` 大概率为假 (False)。
 * @details 编译器会将 `if` 分支的代码放到汇编指令的末尾甚至冷区(Cold Section)，避免污染 I-Cache。
 * 通常用于极其罕见的错误检查、异常抛出或边界条件拦截。
 */

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

#endif  // __ORIGIN_DL_BRANCH_PREDICTION_H__