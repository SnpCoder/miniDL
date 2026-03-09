#include "../../include/optim/sgd.h"

#include "../../include/utils/exception.h"

#ifdef USE_CUDA
// 注意：确保你的头文件路径是对的
#include "../../include/ops/cuda/optim/sgd_cuda.cuh"
#endif

namespace miniDL {
namespace optim {

void SGD::step() {
    for (auto& p : _parameters) {
        if (p.defined() && p.grad().defined()) {
            Tensor grad           = p.grad();
            size_t total_elements = p.element_num();

            // 动量缓冲区初始化
            float* buf_ptr = nullptr;
            if (_momentum != 0.0f) {
                // 使用参数对象的地址作为 Key
                auto it = _momentum_buffers.find(&p);
                if (it == _momentum_buffers.end()) {
                    TensorOptions buf_opts = p.options();
                    buf_opts.requiresGrad(false);  // 缓冲区绝对不能追踪梯度
                    _momentum_buffers[&p] = Tensor::zeros(p.shape(), buf_opts);
                }
                buf_ptr = _momentum_buffers[&p].data_ptr<float>();
            }

            if (p.device().isCuda()) {
#ifdef USE_CUDA
                cuda::launch_sgd_update_float32(p.data_ptr<float>(), grad.data_ptr<float>(),
                                                buf_ptr, _lr, _momentum, _weight_decay,
                                                total_elements);
#else
                MINIDL_THROW_RUNTIME("Framework compiled without CUDA support!");
#endif
            } else {
                // ============================================================
                // CPU 极速原地更新分支 (In-place)
                // ============================================================
                float* p_ptr       = p.data_ptr<float>();
                const float* g_ptr = grad.data_ptr<float>();

                for (size_t i = 0; i < total_elements; ++i) {
                    float g = g_ptr[i];

                    // 1. Weight Decay (L2 正则化)
                    if (_weight_decay != 0.0f) { g += _weight_decay * p_ptr[i]; }

                    // 2. Momentum (动量累积)
                    if (_momentum != 0.0f && buf_ptr != nullptr) {
                        float buf  = buf_ptr[i] * _momentum + g;
                        buf_ptr[i] = buf;
                        g          = buf;
                    }

                    // 3. 原地更新权重 (彻底绕开 Autograd 计算图)
                    p_ptr[i] -= _lr * g;
                }
            }
        }
    }
}

}  // namespace optim
}  // namespace miniDL