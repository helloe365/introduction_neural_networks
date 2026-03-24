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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def test_model():
    """测试训练好的ResNet模型"""
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试数据
    print("正在加载MNIST测试数据...")
    (_, _), (x_test, t_test) = load_mnist(flatten=False, normalize=True)
    
    # 只取前100个样本进行可视化测试
    x_test = x_test[:100]
    t_test = t_test[:100]
    
    # 转换为PyTorch张量
    x_test = torch.FloatTensor(x_test).to(device)
    t_test = torch.LongTensor(t_test).to(device)
    
    # 创建模型并加载权重
    model = ResNet(num_classes=10).to(device)
    model_path = os.path.join(current_dir, "resnet_mnist.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return
    
    # 设置为评估模式
    model.eval()
    
    # 进行预测
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)
        
        # 计算准确率
        correct = (predicted == t_test).sum().item()
        accuracy = 100 * correct / len(t_test)
        
        print(f"测试样本数量: {len(t_test)}")
        print(f"正确预测数量: {correct}")
        print(f"测试准确率: {accuracy:.2f}%")
    
    # 可视化一些预测结果
    visualize_predictions(x_test[:16], t_test[:16], predicted[:16])

def visualize_predictions(images, true_labels, predicted_labels):
    """可视化预测结果"""
    
    # 转换为numpy数组用于显示
    images = images.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    predicted_labels = predicted_labels.cpu().numpy()
    
    # 创建4x4的子图
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('ResNet手写数字识别结果', fontsize=16)
    
    for i in range(16):
        row = i // 4
        col = i % 4
        
        # 显示图像
        axes[row, col].imshow(images[i, 0], cmap='gray')
        axes[row, col].axis('off')
        
        # 设置标题
        true_label = true_labels[i]
        pred_label = predicted_labels[i]
        
        if true_label == pred_label:
            # 预测正确，用绿色
            color = 'green'
            title = f'预测: {pred_label} ✓'
        else:
            # 预测错误，用红色
            color = 'red'
            title = f'预测: {pred_label}, 实际: {true_label} ✗'
        
        axes[row, col].set_title(title, color=color, fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    result_path = os.path.join(current_dir, "resnet_test_results.png")
    plt.savefig(result_path, dpi=300, bbox_inches='tight')
    print(f"测试结果图片已保存到: {result_path}")
    
    plt.show()

def analyze_model_performance():
    """分析模型性能"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载完整测试数据
    (_, _), (x_test, t_test) = load_mnist(flatten=False, normalize=True)
    
    # 转换为PyTorch张量
    x_test = torch.FloatTensor(x_test).to(device)
    t_test = torch.LongTensor(t_test).to(device)
    
    # 创建模型并加载权重
    model = ResNet(num_classes=10).to(device)
    model_path = os.path.join(current_dir, "resnet_mnist.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("模型文件不存在，无法进行性能分析")
        return
    
    model.eval()
    
    # 分批处理以避免内存问题
    batch_size = 100
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            batch_x = x_test[i:i+batch_size]
            batch_t = t_test[i:i+batch_size]
            
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_t.cpu().numpy())
    
    # 计算每个数字的准确率
    digit_accuracy = {}
    for digit in range(10):
        digit_mask = np.array(all_labels) == digit
        digit_predictions = np.array(all_predictions)[digit_mask]
        digit_labels = np.array(all_labels)[digit_mask]
        
        if len(digit_labels) > 0:
            accuracy = (digit_predictions == digit_labels).mean() * 100
            digit_accuracy[digit] = accuracy
        else:
            digit_accuracy[digit] = 0
    
    # 打印结果
    print("\n" + "="*50)
    print("各数字识别准确率:")
    print("="*50)
    for digit in range(10):
        print(f"数字 {digit}: {digit_accuracy[digit]:.2f}%")
    
    overall_accuracy = (np.array(all_predictions) == np.array(all_labels)).mean() * 100
    print(f"\n总体准确率: {overall_accuracy:.2f}%")
    print("="*50)

if __name__ == '__main__':
    print("ResNet手写数字识别测试")
    print("="*50)
    
    # 基本测试
    test_model()
    
    print("\n" + "="*50)
    print("详细性能分析")
    
    # 性能分析
    analyze_model_performance()
