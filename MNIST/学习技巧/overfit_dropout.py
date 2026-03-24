# coding: utf-8
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 为了再现过拟合，减少学习数据
x_train = x_train[:300]
t_train = t_train[:300]

# 设定Dropout比例
dropout_ratio = 0.2

# 创建两个网络进行对比实验
print("开始训练无Dropout网络...")
network_normal = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                                   output_size=10, use_dropout=False)
trainer_normal = Trainer(network_normal, x_train, t_train, x_test, t_test,
                        epochs=301, mini_batch_size=100,
                        optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=False)
trainer_normal.train()

print("开始训练有Dropout网络...")
network_dropout = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                                    output_size=10, use_dropout=True, dropout_ration=dropout_ratio)
trainer_dropout = Trainer(network_dropout, x_train, t_train, x_test, t_test,
                         epochs=301, mini_batch_size=100,
                         optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=False)
trainer_dropout.train()

# 获取训练结果
train_acc_normal, test_acc_normal = trainer_normal.train_acc_list, trainer_normal.test_acc_list
train_acc_dropout, test_acc_dropout = trainer_dropout.train_acc_list, trainer_dropout.test_acc_list

# 绘制对比图形==========
plt.figure(figsize=(12, 5))

# 左图：无Dropout网络的训练和测试准确率
plt.subplot(1, 2, 1)
x = np.arange(len(train_acc_normal))
plt.plot(x, train_acc_normal, marker='o', label='train', markevery=10, linestyle='-', color='blue', markersize=4)
plt.plot(x, test_acc_normal, marker='s', label='test', markevery=10, linestyle='-', color='red', markersize=4)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.xlim(0, 300)
plt.title("Without Dropout")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# 右图：有Dropout网络的训练和测试准确率
plt.subplot(1, 2, 2)
plt.plot(x, train_acc_dropout, marker='o', label='train', markevery=10, linestyle='-', color='blue', markersize=4)
plt.plot(x, test_acc_dropout, marker='s', label='test', markevery=10, linestyle='-', color='red', markersize=4)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.xlim(0, 300)
plt.title("With Dropout")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
save_path = os.path.join(current_dir, 'overfit_dropout_comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Overfitting and dropout comparison saved as '{save_path}'")

# 打印最终结果对比
print("\n=== 最终结果对比 ===")
print(f"无Dropout网络 - 训练准确率: {train_acc_normal[-1]:.4f}, 测试准确率: {test_acc_normal[-1]:.4f}")
print(f"有Dropout网络 - 训练准确率: {train_acc_dropout[-1]:.4f}, 测试准确率: {test_acc_dropout[-1]:.4f}")
print(f"过拟合程度 (训练-测试准确率差):")
print(f"  无Dropout: {train_acc_normal[-1] - test_acc_normal[-1]:.4f}")
print(f"  有Dropout: {train_acc_dropout[-1] - test_acc_dropout[-1]:.4f}")

plt.show()