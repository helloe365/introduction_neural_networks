# coding: utf-8
import sys, os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from dataset.mnist import load_mnist
from alexnet import AlexNet
from common.trainer import Trainer
from common.visualization_utils import plot_training_results

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 处理花费时间较长的情况下减少数据
x_train, t_train = x_train[:6000], t_train[:6000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20  

# 创建AlexNet网络
# AlexNet 原始架构适用于224x224输入，这里适配28x28的MNIST
network = AlexNet(input_dim=(1, 28, 28), output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,  # 减少batch size以适应更复杂的网络
                  optimizer='Adam', optimizer_param={'lr': 0.001},  # 降低学习率
                  evaluate_sample_num_per_epoch=500)
trainer.train()

# 保存参数
params_path = os.path.join(current_dir, "AlexNet_params.pkl")
network.save_params(params_path)
print(f"Saved AlexNet Parameters to: {params_path}")

# 绘制图形
plot_training_results(
    trainer=trainer,
    plot_type="AlexNet",
    title_prefix="AlexNet",
    auto_save=True,
    save_dir=current_dir,
    filename_prefix="AlexNet_training"
)