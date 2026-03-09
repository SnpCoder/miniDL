#pragma once
#include "activation.h"
#include "layernorm.h"
#include "linear.h"
#include "module.h"
#include "multihead_attention.h"

namespace miniDL {
namespace nn {

class TransformerBlock : public Module {
   private:
    // 注意力模块组件
    std::shared_ptr<LayerNorm> ln_1;
    std::shared_ptr<MultiHeadAttention> attn;

    // MLP (前馈神经网络) 组件
    std::shared_ptr<LayerNorm> ln_2;
    std::shared_ptr<Linear> mlp_fc1;
    std::shared_ptr<GELU> mlp_act;
    std::shared_ptr<Linear> mlp_fc2;

   public:
    TransformerBlock(size_t d_model, size_t num_heads, size_t mlp_ratio = 4,
                     Device device = Device("cpu")) {
        // 1. 初始化 Attention 相关的积木
        ln_1 = std::make_shared<LayerNorm>(Shape({d_model}), 1e-5, device);
        attn = std::make_shared<MultiHeadAttention>(d_model, num_heads, device);

        // 2. 初始化 MLP 相关的积木
        ln_2 = std::make_shared<LayerNorm>(Shape({d_model}), 1e-5, device);

        // MLP 的隐藏层维度通常是 d_model 的 4 倍
        size_t d_ff = d_model * mlp_ratio;
        mlp_fc1     = std::make_shared<Linear>(d_model, d_ff, true, device);
        mlp_act     = std::make_shared<GELU>();
        mlp_fc2     = std::make_shared<Linear>(d_ff, d_model, true, device);

        // 3. 【绝对不能忘】：把子模块注册到计算图和参数树中！
        register_module("ln_1", ln_1);
        register_module("attn", attn);
        register_module("ln_2", ln_2);
        register_module("mlp_fc1", mlp_fc1);
        register_module("mlp_fc2", mlp_fc2);
    }

    Tensor forward(const Tensor& x) {
        size_t B = x.shape()[0];
        size_t S = x.shape()[1];
        size_t D = x.shape()[2];

        // ====================================================================
        // 步骤一：Attention 模块 (Pre-LN + Residual Connection)
        // 数学公式：h = x + MHA(LN1(x))
        // ====================================================================
        Tensor ln1_out  = (*ln_1)(x);
        Tensor attn_out = (*attn)(ln1_out);
        Tensor h        = x + attn_out;  // 残差连接 1

        // ====================================================================
        // 步骤二：MLP 升维提取模块 (Pre-LN + Residual Connection)
        // 数学公式：y = h + MLP(LN2(h))
        // ====================================================================
        Tensor ln2_out = (*ln_2)(h);

        // 因为我们的 Linear 暂时只吃 2D 张量，所以做个平滑的降维打击
        Tensor ln2_flat = ln2_out.reshape(Shape({B * S, D}));

        // MLP 前向：Linear -> GELU -> Linear
        Tensor mlp_hidden   = (*mlp_fc1)(ln2_flat);
        mlp_hidden          = (*mlp_act)(mlp_hidden);
        Tensor mlp_out_flat = (*mlp_fc2)(mlp_hidden);

        // 将 MLP 的输出恢复成 3D 序列形状
        Tensor mlp_out = mlp_out_flat.reshape(Shape({B, S, D}));

        // 残差连接 2
        Tensor out = h + mlp_out;

        return out;
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

}  // namespace nn
}  // namespace miniDL