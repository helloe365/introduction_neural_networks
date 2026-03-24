# 项目：Introduction to Neural Networks

- English: [readme_en.md](readme_en.md)

简要说明
----
这是一个面向学习与教学的神经网络入门代码仓库，包含从数值微分、两层神经网络到卷积网络（含 PyTorch 实现）等示例、工具与训练脚本。适合初学者阅读与动手实践。

主要内容（目录概览）
----
- `check_gpu.py`：简单的 GPU 检查脚本。
- `common/`：核心实现模块，包括网络层、优化器、训练器、工具函数等。
- `dataset/`：数据加载与封装（MNIST）。
- `MNIST/`：大量与 MNIST 相关的示例，包括卷积网络、ResNet、训练/测试脚本等。
- `basic_principle/`：基础原理示例，如梯度计算、激活函数比较等。

环境与依赖
----
- 建议使用 Python 3.8+。
- 常用依赖示例（可自行创建 `requirements.txt`）：

```bash
numpy
matplotlib
torch   # 若运行 PyTorch 示例
torchvision
pillow
```

快速开始
----
1. 克隆仓库到本地。
2. 安装依赖（例如使用 pip）：

```bash
pip install -r requirements.txt
```

3. 运行一个简单示例：

```bash
python check_gpu.py
```

运行 MNIST 训练（示例）：

```bash
python MNIST/卷积神经网络/CNN/train_convnet.py
```

说明与建议
----
- 本仓库包含多种实现风格（从手写 NumPy 实现到 PyTorch 实现），用于对比学习深度学习各层面。
- 若要复现实验，请先固定随机种子、记录超参数并保存权重文件（多数训练脚本会有保存逻辑）。

贡献与联系
----
欢迎提交 issue 或 PR 来改进示例、修复 bug 或补充教程内容。如需帮助，请在 issue 中描述你的运行环境与错误信息。

许可证
----
请在提交前与仓库维护者确认许可信息（当前仓库未显式包含 LICENSE 文件）。
