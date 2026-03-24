# coding: utf-8
import sys, os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class AlexNet:
    """AlexNet 卷积神经网络
    
    AlexNet架构 (适配28x28输入):
    conv1 - relu - pool1 - conv2 - relu - pool2 - conv3 - relu - conv4 - relu - conv5 - relu - pool3 - 
    fc1 - relu - dropout - fc2 - relu - dropout - fc3 - softmax
    
    Parameters
    ----------
    input_dim : 输入维度 (通道数, 高度, 宽度)
    output_size : 输出大小（分类数）
    weight_init_std : 权重初始化的标准差
    """
    def __init__(self, input_dim=(1, 28, 28), output_size=10, weight_init_std=0.01):
        # 输入维度
        input_channels, input_height, input_width = input_dim
        
        # 计算各层输出尺寸 (适配28x28输入)
        # Conv1: 32个3x3卷积核，步长1，填充1 -> 28x28
        conv1_output_size = 28
        pool1_output_size = conv1_output_size // 2  # 14x14
        
        # Conv2: 64个3x3卷积核，步长1，填充1 -> 14x14
        conv2_output_size = 14
        pool2_output_size = conv2_output_size // 2  # 7x7
        
        # Conv3: 128个3x3卷积核，步长1，填充1 -> 7x7
        # Conv4: 128个3x3卷积核，步长1，填充1 -> 7x7
        # Conv5: 64个3x3卷积核，步长1，填充1 -> 7x7
        pool3_output_size = 7 // 2  # 3x3 (向下取整)
        
        # 全连接层的输入大小
        fc_input_size = 64 * pool3_output_size * pool3_output_size  # 64*3*3=576
        
        # 初始化权重
        self.params = {}
        
        # Conv1层: 输入1通道，输出32通道，3x3卷积核
        self.params['W1'] = weight_init_std * np.random.randn(32, input_channels, 3, 3)
        self.params['b1'] = np.zeros(32)
        
        # Conv2层: 输入32通道，输出64通道，3x3卷积核
        self.params['W2'] = weight_init_std * np.random.randn(64, 32, 3, 3)
        self.params['b2'] = np.zeros(64)
        
        # Conv3层: 输入64通道，输出128通道，3x3卷积核
        self.params['W3'] = weight_init_std * np.random.randn(128, 64, 3, 3)
        self.params['b3'] = np.zeros(128)
        
        # Conv4层: 输入128通道，输出128通道，3x3卷积核
        self.params['W4'] = weight_init_std * np.random.randn(128, 128, 3, 3)
        self.params['b4'] = np.zeros(128)
        
        # Conv5层: 输入128通道，输出64通道，3x3卷积核
        self.params['W5'] = weight_init_std * np.random.randn(64, 128, 3, 3)
        self.params['b5'] = np.zeros(64)
        
        # FC1层: 全连接层 576 -> 512
        self.params['W6'] = weight_init_std * np.random.randn(fc_input_size, 512)
        self.params['b6'] = np.zeros(512)
        
        # FC2层: 全连接层 512 -> 256
        self.params['W7'] = weight_init_std * np.random.randn(512, 256)
        self.params['b7'] = np.zeros(256)
        
        # FC3层: 全连接层 256 -> output_size
        self.params['W8'] = weight_init_std * np.random.randn(256, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        
        # Conv1 + ReLU + Pool1
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride=1, pad=1)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        # Conv2 + ReLU + Pool2
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], stride=1, pad=1)
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        # Conv3 + ReLU
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], stride=1, pad=1)
        self.layers['Relu3'] = Relu()
        
        # Conv4 + ReLU
        self.layers['Conv4'] = Convolution(self.params['W4'], self.params['b4'], stride=1, pad=1)
        self.layers['Relu4'] = Relu()
        
        # Conv5 + ReLU + Pool3
        self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'], stride=1, pad=1)
        self.layers['Relu5'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        # FC1 + ReLU + Dropout
        self.layers['Affine1'] = Affine(self.params['W6'], self.params['b6'])
        self.layers['Relu6'] = Relu()
        self.layers['Dropout1'] = Dropout(0.5)
        
        # FC2 + ReLU + Dropout
        self.layers['Affine2'] = Affine(self.params['W7'], self.params['b7'])
        self.layers['Relu7'] = Relu()
        self.layers['Dropout2'] = Dropout(0.5)
        
        # FC3 (输出层)
        self.layers['Affine3'] = Affine(self.params['W8'], self.params['b8'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        """求损失函数
        参数x是输入数据、t是教师标签
        """
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """求梯度（数值微分）"""
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in range(1, 9):
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
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W4'], grads['b4'] = self.layers['Conv4'].dW, self.layers['Conv4'].db
        grads['W5'], grads['b5'] = self.layers['Conv5'].dW, self.layers['Conv5'].db
        grads['W6'], grads['b6'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W7'], grads['b7'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W8'], grads['b8'] = self.layers['Affine3'].dW, self.layers['Affine3'].db

        return grads

    def save_params(self, file_name="AlexNet_params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="AlexNet_params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        # 更新层的参数
        layer_mapping = {
            'Conv1': 'W1', 'Conv2': 'W2', 'Conv3': 'W3', 'Conv4': 'W4', 'Conv5': 'W5',
            'Affine1': 'W6', 'Affine2': 'W7', 'Affine3': 'W8'
        }
        
        for layer_name, param_name in layer_mapping.items():
            if hasattr(self.layers[layer_name], 'W'):
                self.layers[layer_name].W = self.params[param_name]
                self.layers[layer_name].b = self.params[param_name.replace('W', 'b')]
