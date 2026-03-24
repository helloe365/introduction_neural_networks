# coding: utf-8
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer
from common.visualization_utils import plot_training_results

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 处理花费时间较长的情况下减少数据
x_train, t_train = x_train[:6000], t_train[:6000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 保存参数
params_path = os.path.join(current_dir, "params.pkl")
network.save_params(params_path)
print(f"Saved Network Parameters to: {params_path}")

# 绘制图形并自动保存到当前目录
plot_training_results(
    trainer=trainer,
    plot_type="cnn",
    title_prefix="SimpleConvNet",
    auto_save=True,
    save_dir=current_dir,
    filename_prefix="convnet_training"
)
