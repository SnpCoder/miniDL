#pragma once
#include <cmath>
#include <stdexcept>

#include "../ops/math/add.h"
#include "../ops/math/mul.h"
#include "../ops/nn/causal_mask.h"
#include "activation.h"  // 包含 Softmax
#include "linear.h"
#include "module.h"

namespace miniDL {
namespace nn {

class MultiHeadAttention : public Module {
   private:
    size_t _d_model;
    size_t _num_heads;
    size_t _d_head;

    // 四个全连接层，负责高维空间映射
    std::shared_ptr<Linear> q_proj;
    std::shared_ptr<Linear> k_proj;
    std::shared_ptr<Linear> v_proj;
    std::shared_ptr<Linear> out_proj;

    // Softmax 负责把 Attention Score 转化为概率分布
    std::shared_ptr<Softmax> softmax;

   public:
    MultiHeadAttention(size_t d_model, size_t num_heads, Device device = Device("cpu"))
        : _d_model(d_model), _num_heads(num_heads) {
        if (d_model % num_heads != 0) {
            throw std::invalid_argument("d_model must be exactly divisible by num_heads");
        }
        _d_head = d_model / num_heads;

        // 初始化底层的 Module 积木
        q_proj   = std::make_shared<Linear>(d_model, d_model, true, device);
        k_proj   = std::make_shared<Linear>(d_model, d_model, true, device);
        v_proj   = std::make_shared<Linear>(d_model, d_model, true, device);
        out_proj = std::make_shared<Linear>(d_model, d_model, true, device);
        softmax  = std::make_shared<Softmax>();

        // 【极其重要】：必须把子模块注册到网络中，否则它们收不到优化器的梯度！
        register_module("q_proj", q_proj);
        register_module("k_proj", k_proj);
        register_module("v_proj", v_proj);
        register_module("out_proj", out_proj);
    }

    Tensor forward(const Tensor& x) {
        size_t B  = x.shape()[0];  // Batch size
        size_t S  = x.shape()[1];  // Sequence length
        size_t D  = _d_model;      // Model dimension (e.g., 768)
        size_t H  = _num_heads;    // Number of heads (e.g., 12)
        size_t Dh = _d_head;       // Head dimension (e.g., 64)

        // ====================================================================
        // 1. 全连接映射 (压扁为 2D 交给 Linear 处理)
        // ====================================================================
        Tensor x_flat = x.reshape(Shape({B * S, D}));

        Tensor q_flat = (*q_proj)(x_flat);
        Tensor k_flat = (*k_proj)(x_flat);
        Tensor v_flat = (*v_proj)(x_flat);

        // ====================================================================
        // 2. 张量变形魔法：拆分多头并换位
        // 原理: [B*S, D] -> [B, S, H, Dh] -> [B, H, S, Dh]
        // ====================================================================
        Tensor q = q_flat.reshape(Shape({B, S, H, Dh})).permute({0, 2, 1, 3}).contiguous();
        Tensor k = k_flat.reshape(Shape({B, S, H, Dh})).permute({0, 2, 1, 3}).contiguous();
        Tensor v = v_flat.reshape(Shape({B, S, H, Dh})).permute({0, 2, 1, 3}).contiguous();

        // ====================================================================
        // 3. 降维打击：为 3D BMM 做准备
        // 原理: [B, H, S, Dh] -> [B*H, S, Dh]
        // ====================================================================
        q = q.reshape(Shape({B * H, S, Dh}));
        k = k.reshape(Shape({B * H, S, Dh}));
        v = v.reshape(Shape({B * H, S, Dh}));

        // ====================================================================
        // 4. 核心数学公式：Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        // ====================================================================
        // 4.1 转置 K 矩阵: [B*H, Dh, S] (必须 contiguous 因为被 permute 过)
        Tensor k_t = k.permute({0, 2, 1}).contiguous();

        // 4.2 算注意力分数 S: [B*H, S, S]
        Tensor scores = q.bmm(k_t);

        // 4.3 缩放 (除以 sqrt(d_head))
        float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
        scores      = scores * scale;

        // causal mask
        scores = CausalMaskOp::apply(scores);

        // 4.4 Softmax 概率归一化
        Tensor attn_weights = (*softmax)(scores);

        // 4.5 与 V 矩阵相乘，提取上下文: [B*H, S, Dh]
        Tensor context = attn_weights.bmm(v);

        // ====================================================================
        // 5. 收网阶段：将多头拼接回原来的形状
        // 原理: [B*H, S, Dh] -> [B, H, S, Dh] -> [B, S, H, Dh] -> [B*S, D]
        // ====================================================================
        context             = context.reshape(Shape({B, H, S, Dh}));
        context             = context.permute({0, 2, 1, 3}).contiguous();
        Tensor context_flat = context.reshape(Shape({B * S, D}));

        // ====================================================================
        // 6. 最后的输出线性映射
        // ====================================================================
        Tensor out_flat = (*out_proj)(context_flat);

        // 还原为最终的 [Batch, Seq, D]
        Tensor out = out_flat.reshape(Shape({B, S, D}));

        return out;
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

}  // namespace nn
}  // namespace miniDL