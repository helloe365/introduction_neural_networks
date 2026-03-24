# coding: utf-8
"""
PyTorch implementation of DeepConvNet training script
Converted from ch08/train_deepnet.py

Features:
- CUDA GPU acceleration support
- Adam optimizer with cross-entropy loss
- Training visualization with matplotlib
- Same training configuration as original implementation
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

from dataset.mnist import load_mnist
from deep_convnet_pytorch import DeepConvNetPyTorch, get_device
from common.visualization_utils import plot_training_results


def create_data_loaders(x_train, t_train, x_test, t_test, batch_size=100):
    """创建PyTorch数据加载器"""
    # 转换为PyTorch张量
    x_train_tensor = torch.FloatTensor(x_train)
    t_train_tensor = torch.LongTensor(t_train)
    x_test_tensor = torch.FloatTensor(x_test)
    t_test_tensor = torch.LongTensor(t_test)
    
    # 创建数据集
    train_dataset = TensorDataset(x_train_tensor, t_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, t_test_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def calculate_accuracy(model, data_loader, device):
    """计算模型在数据集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return correct / total


def train_model(model, train_loader, test_loader, device, epochs=20, lr=0.001):
    """训练模型"""
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练过程
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    print("Starting training...")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Learning Rate: {lr}")
    print("-" * 50)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        # 训练一个epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 记录每个batch的损失
            train_loss_list.append(loss.item())
        
        # 计算准确率
        train_acc = calculate_accuracy(model, train_loader, device)
        test_acc = calculate_accuracy(model, test_loader, device)
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        
        # 打印训练信息
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Test Acc: {test_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print("-" * 50)
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Final Test Accuracy: {test_acc_list[-1]:.4f}")
    
    return train_loss_list, train_acc_list, test_acc_list


def plot_pytorch_training_results(train_loss_list, train_acc_list, test_acc_list, current_dir):
    """绘制PyTorch DeepConvNet训练结果可视化图表"""
    # 使用可复用的可视化工具绘制训练结果并自动保存到当前目录
    from common.visualization_utils import plot_training_results

    plot_training_results(
        train_loss_list=train_loss_list,
        train_acc_list=train_acc_list,
        test_acc_list=test_acc_list,
        plot_type="cnn",
        title_prefix="DeepConvNet-PyTorch",
        auto_save=True,
        save_dir=current_dir,
        filename_prefix="pytorch_training"
    )


def main():
    """主函数"""
    print("PyTorch DeepConvNet Training")
    print("=" * 50)
    
    # 检测设备
    device = get_device()
    
    # 加载MNIST数据集
    print("\nLoading MNIST dataset...")
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    
    # 为了快速测试，减少数据量 (与原始实现保持一致)
    # x_train, t_train = x_train[:6000], t_train[:6000]
    # x_test, t_test = x_test[:1000], t_test[:1000]

    max_epochs = 20
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {t_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {t_test.shape}")
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        x_train, t_train, x_test, t_test, batch_size=100
    )
    
    # 创建模型
    print("\nCreating DeepConvNet model...")
    model = DeepConvNetPyTorch(input_dim=(1, 28, 28), hidden_size=50, output_size=10)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # 训练模型
    print("\n" + "=" * 50)
    train_loss_list, train_acc_list, test_acc_list = train_model(
        model, train_loader, test_loader, device, epochs=max_epochs, lr=0.001
    )
    
    # 保存模型参数
    print("\nSaving model parameters...")
    model_path = os.path.join(current_dir, "deep_convnet_pytorch.pth")
    model.save_params(model_path)
    
    # 绘制训练结果
    print("\nPlotting training results...")
    plot_pytorch_training_results(train_loss_list, train_acc_list, test_acc_list, current_dir)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
