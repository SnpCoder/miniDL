#include <getopt.h>

#include <cstdlib>
#include <vector>

#include "../../include/core/tensor.h"
#include "../../include/nn/linear.h"
#include "../../include/nn/loss.h"
#include "../../include/ops/math/add.h"
#include "../../include/ops/math/mul.h"
#include "../../include/optim/sgd.h"
#include "../../include/utils/log.h"

using namespace miniDL;

void usage(const char* program_name) {
    MINIDL_PRINT("Usage: {} [--cpu] [-d device_id] [-h]", program_name);
    MINIDL_PRINT("  --cpu            Use CPU (overrides auto/CUDA)");
    MINIDL_PRINT(
        "  -d, --device     Device: -1 for CPU, >= 0 for GPU id. Omit for auto (CUDA if "
        "available)");
    MINIDL_PRINT("  -h, --help       Show this help message");
}

int main(int argc, char** argv) {
    // ========================================================================
    // 1. 命令行解析与设备选择 (吸收自 Origin 框架的优雅设计)
    // ========================================================================
    int device_id                       = -2;  // -2 = auto, -1 = CPU, >= 0 = GPU
    static struct option long_options[] = {{"cpu", no_argument, 0, 'c'},
                                           {"device", required_argument, 0, 'd'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "cd:h", long_options, &option_index)) != -1) {
        switch (c) {
            case 'c':
                device_id = -1;
                break;
            case 'd':
                device_id = std::atoi(optarg);
                break;
            case 'h':
                usage(argv[0]);
                std::exit(0);
            case '?':
                MINIDL_PRINT("Use -h or --help for usage information");
                std::exit(1);
            default:
                break;
        }
    }

    Device target_device(Device("cpu"));
    if (device_id == -2) {
#ifdef USE_CUDA
        device_id = 0;  // Auto: 优先使用 GPU
#else
        device_id = -1;
#endif
    }

    if (device_id >= 0) {
#ifdef USE_CUDA
        target_device = Device(std::string("cuda:") + std::to_string(device_id));
#else
        loge("CUDA is not available on this system but GPU was requested!");
        return 1;
#endif
    }

    MINIDL_PRINT("====================================================");
    MINIDL_PRINT("🚀 miniDL 框架首秀：一元线性回归");
    MINIDL_PRINT("🎯 目标任务：拟合直线方程 y = 3.0 * x + 5.0 + noise");
    MINIDL_PRINT("💻 Use Device: {}", target_device.to_string());
    MINIDL_PRINT("====================================================");

    // ========================================================================
    // 2. 准备数据 (加入噪声，模拟真实世界)
    // ========================================================================
    size_t input_size = 100;
    auto opt          = device(target_device).requiresGrad(false);

    // 生成输入 X (均匀分布，假设在 -10 到 10 之间)
    auto x = Tensor::uniform(Shape({input_size, 1}), -10.0f, 10.0f, opt);

    // 生成噪声 Noise (正态分布 N(0, 1) 乘以 0.5 缩放)
    auto noise = Tensor::randn(Shape({input_size, 1}), 0.0f, 1.0f, opt);

    // 真实分布 Y = 3x + 5 + noise
    // (注意：这里我们可以直接用 + 和 *，因为 requiresGrad=false，不会构建计算图)
    auto target_y = x * 3.0f;
    target_y      = target_y + noise * 0.5f;

    // 我们手动给 target 加上偏置 5.0 (这里假定你的算子能处理标量加法，
    // 如果还没写标量加法，可以用 uniform 全 5.0 的 Tensor 代替)
    Tensor bias_tensor = Tensor::ones(Shape({input_size, 1}), opt);
    bias_tensor        = bias_tensor * 5.0f;
    target_y           = target_y + bias_tensor;

    // ========================================================================
    // 3. 构建高层神经网络与优化器 (PyTorch 架构的降维打击)
    // ========================================================================
    // 输入 1 维，输出 1 维，开启偏置
    nn::Linear model(1, 1, true, target_device);
    nn::MSELoss criterion;
    // SGD 优化器，学习率 0.01
    optim::SGD optimizer(model.parameters(), 0.01f);

    // ========================================================================
    // 4. 标准训练循环 (Training Loop)
    // ========================================================================
    int iters = 1000;

    for (int i = 0; i <= iters; i++) {
        // [1] 清零梯度
        optimizer.zero_grad();

        // [2] 前向传播
        auto y_pred = model(x);

        // [3] 计算损失
        auto loss = criterion(y_pred, target_y);

        // [4] 反向传播
        // 手动为 loss 的源头注入梯度 1.0
        Tensor grad_out = Tensor::ones(Shape({1}), device(target_device));
        loss.impl()->set_grad(grad_out.shared_impl());
        loss.backward();

        // [5] 优化器原地更新参数！安全、无碎片！
        optimizer.step();

        // [6] 打印日志
        if (i % 20 == 0 || i == iters) {
            float loss_val = loss.to(Device("cpu")).data_ptr<float>()[0];

            auto params = model.parameters();
            // 注意之前的排序：我们现在用的是 Vector，按注册顺序，先 weight 后 bias
            float w_val = params[0].to(Device("cpu")).data_ptr<float>()[0];
            float b_val = params[1].to(Device("cpu")).data_ptr<float>()[0];

            MINIDL_PRINT("iter {}: loss = {:.4f}, w = {:.4f}, b = {:.4f}", i, loss_val, w_val,
                         b_val);
        }
    }

    MINIDL_PRINT("🎉 训练完成！权重已无限逼近 w=3.0, b=5.0");
    return 0;
}