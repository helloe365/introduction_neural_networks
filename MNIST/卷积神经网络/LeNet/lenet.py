# coding: utf-8
import sys, os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class LeNet:
    """LeNet-5 卷积神经网络
    
    经典的LeNet架构：
    conv1 - sigmoid - pool1 - conv2 - sigmoid - pool2 - affine1 - sigmoid - affine2 - sigmoid - affine3 - softmax
    
    Parameters
    ----------
    input_dim : 输入维度 (通道数, 高度, 宽度)
    output_size : 输出大小（分类数）
    weight_init_std : 权重初始化的标准差
    """
    def __init__(self, input_dim=(1, 28, 28), output_size=10, weight_init_std=0.01):
        # 输入维度
        input_size = input_dim[1]
        
        # LeNet-5 的网络参数
        # Conv1: 6个5x5卷积核，步长1，无填充
        conv1_output_size = (input_size - 5) + 1  # 28-5+1=24
        pool1_output_size = conv1_output_size // 2  # 24//2=12
        
        # Conv2: 16个5x5卷积核，步长1，无填充
        conv2_output_size = (pool1_output_size - 5) + 1  # 12-5+1=8
        pool2_output_size = conv2_output_size // 2  # 8//2=4
        
        # 全连接层的输入大小
        fc_input_size = 16 * pool2_output_size * pool2_output_size  # 16*4*4=256
        
        # 初始化权重
        self.params = {}
        
        # Conv1层: 输入1通道，输出6通道，5x5卷积核
        self.params['W1'] = weight_init_std * np.random.randn(6, input_dim[0], 5, 5)
        self.params['b1'] = np.zeros(6)
        
        # Conv2层: 输入6通道，输出16通道，5x5卷积核
        self.params['W2'] = weight_init_std * np.random.randn(16, 6, 5, 5)
        self.params['b2'] = np.zeros(16)
        
        # FC1层: 全连接层 256 -> 120
        self.params['W3'] = weight_init_std * np.random.randn(fc_input_size, 120)
        self.params['b3'] = np.zeros(120)
        
        # FC2层: 全连接层 120 -> 84
        self.params['W4'] = weight_init_std * np.random.randn(120, 84)
        self.params['b4'] = np.zeros(84)
        
        # FC3层: 全连接层 84 -> output_size
        self.params['W5'] = weight_init_std * np.random.randn(84, output_size)
        self.params['b5'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        
        # Conv1 + Sigmoid + Pool1
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride=1, pad=0)
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        # Conv2 + Sigmoid + Pool2
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], stride=1, pad=0)
        self.layers['Sigmoid2'] = Sigmoid()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        # FC1 + Sigmoid
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Sigmoid3'] = Sigmoid()
        
        # FC2 + Sigmoid
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Sigmoid4'] = Sigmoid()
        
        # FC3 (输出层)
        self.layers['Affine3'] = Affine(self.params['W5'], self.params['b5'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """求损失函数
        参数x是输入数据、t是教师标签
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """求梯度（数值微分）"""
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3, 4, 5):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """求梯度（误差反向传播法）"""
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W5'], grads['b5'] = self.layers['Affine3'].dW, self.layers['Affine3'].db

        return grads

    def save_params(self, file_name="LeNet_params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="LeNet_params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        # 更新层的参数
        layer_param_mapping = [
            ('Conv1', 'W1', 'b1'),
            ('Conv2', 'W2', 'b2'),
            ('Affine1', 'W3', 'b3'),
            ('Affine2', 'W4', 'b4'),
            ('Affine3', 'W5', 'b5')
        ]

        for layer_name, w_key, b_key in layer_param_mapping:
            if hasattr(self.layers[layer_name], 'W'):
                self.layers[layer_name].W = self.params[w_key]
                self.layers[layer_name].b = self.params[b_key]
