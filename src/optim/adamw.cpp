#include "../../include/optim/adamw.h"

#include "../../include/utils/exception.h"

#ifdef USE_CUDA
#include "../../include/ops/cuda/optim/adamw_cuda.cuh"
#endif

namespace miniDL {
namespace optim {

void AdamW::zero_grad() {
    for (auto& p : _parameters) {
        if (p.defined() && p.grad().defined()) { p.impl()->set_grad(nullptr); }
    }
}

void AdamW::step() {
    _step_count++;

    // 计算当前的偏差校正系数 (随着 step 增加，趋近于 1)
    float beta1_t = std::pow(_beta1, _step_count);
    float beta2_t = std::pow(_beta2, _step_count);

    for (auto& p : _parameters) {
        if (!p.defined() || !p.grad().defined()) continue;

        Tensor grad = p.grad();
        void* key   = p.impl();  // 获取底层唯一标识

        // 懒加载：如果 m 和 v 是第一次用到，给它们分配全 0 的内存
        if (_m.find(key) == _m.end()) {
            _m[key] = Tensor::zeros(p.shape(), p.options());
            _v[key] = Tensor::zeros(p.shape(), p.options());
        }

        Tensor& m = _m[key];
        Tensor& v = _v[key];
        size_t N  = p.element_num();

        if (p.device().isCpu()) {
            float* w_ptr       = p.data_ptr<float>();
            const float* g_ptr = grad.data_ptr<float>();
            float* m_ptr       = m.data_ptr<float>();
            float* v_ptr       = v.data_ptr<float>();

            // 极速 CPU 融合更新循环
            for (size_t i = 0; i < N; ++i) {
                float weight = w_ptr[i];
                float g      = g_ptr[i];

                weight = weight - _lr * _weight_decay * weight;

                float m_val = _beta1 * m_ptr[i] + (1.0f - _beta1) * g;
                float v_val = _beta2 * v_ptr[i] + (1.0f - _beta2) * g * g;
                m_ptr[i]    = m_val;
                v_ptr[i]    = v_val;

                float m_hat = m_val / (1.0f - beta1_t);
                float v_hat = v_val / (1.0f - beta2_t);

                w_ptr[i] = weight - _lr * (m_hat / (std::sqrt(v_hat) + _eps));
            }
        } else if (p.device().isCuda()) {
#ifdef USE_CUDA
            cuda::launch_adamw_step_float32(p.data_ptr<float>(), grad.data_ptr<float>(),
                                            m.data_ptr<float>(), v.data_ptr<float>(), _lr, _beta1,
                                            _beta2, _eps, _weight_decay, beta1_t, beta2_t, N);
#endif
        }
    }
}

}  // namespace optim
}  // namespace miniDL