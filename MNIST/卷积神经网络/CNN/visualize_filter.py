# coding: utf-8
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
from simple_convnet import SimpleConvNet
from common.visualization_utils import visualize_conv_filters


network = SimpleConvNet()
# 随机进行初始化后的权重
print("显示随机初始化后的权重:")
visualize_conv_filters(network, layer_name='W1', title_prefix="Random Initialized")

# 学习后的权重
params_path = os.path.join(current_dir, "params.pkl")
if os.path.exists(params_path):
    network.load_params(params_path)
    print("显示训练后的权重:")
    visualize_conv_filters(network, layer_name='W1', title_prefix="Trained")
else:
    print(f"参数文件不存在: {params_path}")
    print("请先运行 train_convnet.py 来训练模型并生成参数文件")