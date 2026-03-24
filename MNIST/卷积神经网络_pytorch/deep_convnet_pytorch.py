# coding: utf-8
"""
Network architecture:
    conv - relu - conv - relu - pool -
    conv - relu - conv - relu - pool -
    conv - relu - conv - relu - pool -
    affine - relu - dropout - affine - dropout - softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DeepConvNetPyTorch(nn.Module):
    
    def __init__(self, input_dim=(1, 28, 28), hidden_size=50, output_size=10):
        super(DeepConvNetPyTorch, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 卷积层参数 (与原始实现保持一致)
        # conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1}
        # conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1}
        # conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1}
        # conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1}
        # conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1}
        # conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1}
        
        # 第一组卷积层: conv - relu - conv - relu - pool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二组卷积层: conv - relu - conv - relu - pool
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三组卷积层: conv - relu - conv - relu - pool
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算卷积层输出后的特征图大小
        # 输入: 28x28 -> conv1,conv2 -> 28x28 -> pool1 -> 14x14
        # -> conv3,conv4 -> 14x14 -> pool2 -> 7x7  
        # -> conv5,conv6 -> 7x7 -> pool3 -> 3x3 (由于padding=2在conv4)
        # 实际上conv4的padding=2会让特征图变大，需要仔细计算
        
        # 根据原始实现，最终特征图大小应该是4x4
        # 这里我们使用自适应池化来确保输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 全连接层: affine - relu - dropout - affine - dropout
        self.fc1 = nn.Linear(64 * 4 * 4, hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout2 = nn.Dropout(0.5)
        
        # 初始化权重 (使用He初始化，适合ReLU)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用He初始化权重 (适合ReLU激活函数)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 第一组: conv - relu - conv - relu - pool
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # 第二组: conv - relu - conv - relu - pool
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # 第三组: conv - relu - conv - relu - pool
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        
        # 自适应池化确保输出大小为4x4
        x = self.adaptive_pool(x)
        
        # 展平特征图
        x = x.view(x.size(0), -1)  # (batch_size, 64*4*4)
        
        # 全连接层: affine - relu - dropout - affine - dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        
        return x
    
    def predict(self, x):
        """预测函数 (用于推理)"""
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs.data, 1)
        return predicted
    
    def save_params(self, file_name="deep_convnet_pytorch.pth"):
        """保存模型参数"""
        torch.save(self.state_dict(), file_name)
        print(f"Model parameters saved to {file_name}")
    
    def load_params(self, file_name="deep_convnet_pytorch.pth"):
        """加载模型参数"""
        self.load_state_dict(torch.load(file_name))
        print(f"Model parameters loaded from {file_name}")


def get_device():
    """检测并返回可用的设备 (CUDA GPU 或 CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
    
    return device

