# coding: utf-8
import sys, os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（需要回到两级父目录）
project_root = os.path.dirname(os.path.dirname(current_dir))
# 将项目根目录添加到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataset.mnist import load_mnist
from common.visualization_utils import plot_training_results

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")

class ResidualBlock(nn.Module):
    """残差块实现"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    """ResNet网络实现"""
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # 残差块
        self.layer1 = ResidualBlock(16, 16, stride=1)
        self.layer2 = ResidualBlock(16, 32, stride=2)  # 14x14 -> 7x7
        self.layer3 = ResidualBlock(32, 64, stride=1)
        self.layer4 = ResidualBlock(64, 64, stride=1)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x



# 读入数据
print("正在加载MNIST数据集...")
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True)

# 减少数据量以加快训练（可选）
# x_train, t_train = x_train[:6000], t_train[:6000]
# x_test, t_test = x_test[:1000], t_test[:1000]

# 训练参数
num_epochs = 20
batch_size = 100

print(f"训练数据形状: {x_train.shape}")
print(f"测试数据形状: {x_test.shape}")

# 转换为PyTorch张量
x_train = torch.FloatTensor(x_train).to(device)
t_train = torch.LongTensor(t_train).to(device)
x_test = torch.FloatTensor(x_test).to(device)
t_test = torch.LongTensor(t_test).to(device)

# 创建数据加载器
train_dataset = TensorDataset(x_train, t_train)
test_dataset = TensorDataset(x_test, t_test)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型
model = ResNet(num_classes=10).to(device)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)



# 记录训练过程
train_loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()

    # 计算训练准确率
    train_acc = correct_train / total_train
    avg_loss = running_loss / len(train_loader)

    # 测试阶段
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()

    test_acc = correct_test / total_test

    # 记录结果
    train_loss_list.append(avg_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    print(f"loss:{avg_loss:.4f}, train acc:{train_acc:.4f}, test acc:{test_acc:.4f}")

# 保存模型
model_path = os.path.join(current_dir, "resnet_mnist.pth")
torch.save(model.state_dict(), model_path)
print(f"模型已保存到: {model_path}")

# 绘制图形并自动保存到当前目录
plot_training_results(
    train_loss_list=train_loss_list,
    train_acc_list=train_acc_list,
    test_acc_list=test_acc_list,
    plot_type="cnn",
    title_prefix="ResNet",
    auto_save=True,
    save_dir=current_dir,
    filename_prefix="resnet_training"
)
