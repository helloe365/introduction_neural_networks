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
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# 0:读入MNIST数据==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 1:进行实验的设置==========
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()
#optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
train_acc = {}
test_acc = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []
    train_acc[key] = []
    test_acc[key] = []


# 2:开始训练==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    # 每100次迭代计算一次准确率
    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            # 计算训练准确率（使用小批量数据以节省时间）
            train_accuracy = networks[key].accuracy(x_batch, t_batch)
            # 计算测试准确率（使用前1000个测试样本以节省时间）
            test_accuracy = networks[key].accuracy(x_test[:1000], t_test[:1000])

            train_acc[key].append(train_accuracy)
            test_acc[key].append(test_accuracy)

            print(key + " - loss:" + str(loss) + ", train acc:" + str(train_accuracy) + ", test acc:" + str(test_accuracy))


# 3.绘制图形==========
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}

# 创建包含两个子图的图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 绘制损失曲线
x_loss = np.arange(max_iterations)
for key in optimizers.keys():
    ax1.plot(x_loss, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
ax1.set_xlabel("iterations")
ax1.set_ylabel("loss")
ax1.set_ylim(0, 1)
ax1.set_title("Training Loss Comparison")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 绘制准确率曲线
x_acc = np.arange(0, max_iterations, 100)  # 每100次迭代记录一次准确率
for key in optimizers.keys():
    ax2.plot(x_acc, train_acc[key], marker=markers[key], markevery=2, label=key + ' (train)', linestyle='-')
    ax2.plot(x_acc, test_acc[key], marker=markers[key], markevery=2, label=key + ' (test)', linestyle='--', alpha=0.7)

ax2.set_xlabel("iterations")
ax2.set_ylabel("accuracy")
ax2.set_ylim(0, 1)
ax2.set_title("Accuracy Comparison")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图像
save_path = os.path.join(current_dir, 'optimizer_comparison_mnist.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Optimizer comparison on MNIST saved as '{save_path}'")

plt.show()
