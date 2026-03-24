# coding: utf-8
import sys, os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset.mnist import load_mnist
from train_ResNet import ResNet

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def demo_resnet_architecture():
    """演示ResNet网络架构"""
    
    print("="*60)
    print("ResNet手写数字识别网络架构演示")
    print("="*60)
    
    # 创建模型
    model = ResNet(num_classes=10)
    
    # 打印模型结构
    print("\n网络结构:")
    print("-"*40)
    print(model)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    print(f"\n前向传播测试:")
    print("-"*40)
    
    # 创建一个随机输入
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    print(f"输入张量形状: {input_tensor.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        print(f"输出张量形状: {output.shape}")
        
        # 应用softmax获得概率
        probabilities = torch.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        print(f"预测结果: {predictions.numpy()}")
        print(f"预测概率形状: {probabilities.shape}")

def demo_single_prediction():
    """演示单张图片的预测过程"""
    
    print("\n" + "="*60)
    print("单张图片预测演示")
    print("="*60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载一张测试图片
    (_, _), (x_test, t_test) = load_mnist(flatten=False, normalize=True)
    
    # 选择第一张图片
    test_image = x_test[0:1]  # 保持batch维度
    true_label = t_test[0]
    
    print(f"测试图片形状: {test_image.shape}")
    print(f"真实标签: {true_label}")
    
    # 转换为PyTorch张量
    test_tensor = torch.FloatTensor(test_image).to(device)
    
    # 创建模型
    model = ResNet(num_classes=10).to(device)
    
    # 尝试加载训练好的权重
    model_path = os.path.join(current_dir, "resnet_mnist.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("已加载训练好的模型权重")
    else:
        print("使用随机初始化的权重（未训练）")
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(test_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_label].item()
    
    print(f"预测标签: {predicted_label}")
    print(f"预测置信度: {confidence:.4f}")
    print(f"预测是否正确: {'是' if predicted_label == true_label else '否'}")
    
    # 显示图片和预测结果
    plt.figure(figsize=(8, 6))
    
    # 显示原图
    plt.subplot(1, 2, 1)
    plt.imshow(test_image[0, 0], cmap='gray')
    plt.title(f'原始图片\n真实标签: {true_label}')
    plt.axis('off')
    
    # 显示预测概率分布
    plt.subplot(1, 2, 2)
    prob_array = probabilities[0].cpu().numpy()
    bars = plt.bar(range(10), prob_array)
    
    # 高亮预测的类别
    bars[predicted_label].set_color('red')
    
    plt.title(f'预测概率分布\n预测: {predicted_label} (置信度: {confidence:.3f})')
    plt.xlabel('数字类别')
    plt.ylabel('概率')
    plt.xticks(range(10))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    demo_path = os.path.join(current_dir, "resnet_demo.png")
    plt.savefig(demo_path, dpi=300, bbox_inches='tight')
    print(f"演示图片已保存到: {demo_path}")
    
    plt.show()

def compare_with_without_residual():
    """比较有无残差连接的效果（理论说明）"""
    
    print("\n" + "="*60)
    print("残差连接的重要性")
    print("="*60)
    
    print("""
残差网络(ResNet)的核心创新是引入了残差连接(Skip Connection)：

1. 传统深度网络问题:
   - 梯度消失：随着网络加深，梯度在反向传播中逐渐消失
   - 退化问题：网络越深，训练误差反而增加
   
2. 残差连接解决方案:
   - 跳跃连接：F(x) + x，直接将输入加到输出
   - 学习残差：网络学习F(x) = H(x) - x，而不是直接学习H(x)
   - 梯度直通：梯度可以直接通过跳跃连接传播
   
3. 数学原理:
   - 传统: y = F(x)
   - ResNet: y = F(x) + x
   - 如果最优映射接近恒等映射，F(x)只需学习接近0的残差
   
4. 实际效果:
   - 允许训练更深的网络
   - 加速收敛
   - 提高准确率
   - 缓解梯度消失问题
    """)

def show_training_tips():
    """显示训练技巧和建议"""
    
    print("\n" + "="*60)
    print("ResNet训练技巧和建议")
    print("="*60)
    
    print("""
1. 数据预处理:
   - 归一化：将像素值缩放到[0,1]或[-1,1]
   - 数据增强：旋转、平移、缩放等（可选）
   
2. 网络初始化:
   - He初始化：适用于ReLU激活函数
   - Xavier初始化：适用于Sigmoid/Tanh激活函数
   
3. 优化器选择:
   - Adam：自适应学习率，收敛快
   - SGD+Momentum：经典选择，需要调参
   - 学习率调度：StepLR, CosineAnnealingLR等
   
4. 正则化技术:
   - Batch Normalization：加速训练，提高稳定性
   - Dropout：防止过拟合（在全连接层使用）
   - Weight Decay：L2正则化
   
5. 训练策略:
   - 渐进式训练：先用小数据集验证
   - 早停：监控验证集性能
   - 模型保存：保存最佳模型权重
   
6. 超参数调优:
   - 学习率：通常从0.001开始
   - 批次大小：32-256，根据显存调整
   - 训练轮数：根据收敛情况调整
    """)

if __name__ == '__main__':
    # 网络架构演示
    demo_resnet_architecture()
    
    # 单张图片预测演示
    demo_single_prediction()
    
    # 残差连接重要性说明
    compare_with_without_residual()
    
    # 训练技巧
    show_training_tips()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
