# coding: utf-8
"""
PyTorch DeepConvNet 测试和评估脚本

这个脚本专门用于加载已训练的模型并进行测试评估，不进行重新训练。

主要功能:
- 加载训练好的模型 (deep_convnet_pytorch.pth)
- 在MNIST测试集上评估性能
- 生成详细的分类报告和混淆矩阵
- 可视化样本预测结果
- 分析预测错误的样本
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from dataset.mnist import load_mnist
from deep_convnet_pytorch import DeepConvNetPyTorch, get_device


def load_trained_model(model_path, device):
    """加载训练好的模型"""
    model = DeepConvNetPyTorch(input_dim=(1, 28, 28), hidden_size=50, output_size=10)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found. Using untrained model.")
    
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, x_test, t_test, device, batch_size=100):
    """评估模型性能"""
    model.eval()

    # 转换为PyTorch张量
    x_test_tensor = torch.FloatTensor(x_test).to(device)

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for i in range(0, len(x_test_tensor), batch_size):
            batch_x = x_test_tensor[i:i+batch_size]
            outputs = model(batch_x)

            # 获取预测结果
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # 计算准确率
    accuracy = np.mean(all_predictions == t_test)

    return all_predictions, all_probabilities, accuracy


def plot_sample_predictions(x_test, t_test, predictions, probabilities, current_dir, num_samples=10):
    """可视化样本预测结果"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    # 随机选择样本
    indices = np.random.choice(len(x_test), num_samples, replace=False)

    for i, idx in enumerate(indices):
        # 显示图像
        axes[i].imshow(x_test[idx].reshape(28, 28), cmap='gray')

        # 设置标题
        true_label = t_test[idx]
        pred_label = predictions[idx]
        confidence = probabilities[idx][pred_label]

        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}',
                         color=color, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    save_path = os.path.join(current_dir, 'sample_predictions_pytorch.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Sample predictions saved as '{save_path}'")


def plot_confusion_matrix(t_test, predictions, current_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(t_test, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    save_path = os.path.join(current_dir, 'confusion_matrix_pytorch.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved as '{save_path}'")


def analyze_errors(x_test, t_test, predictions, probabilities, current_dir, num_errors=10):
    """分析错误预测的样本"""
    # 找到错误预测的样本
    error_indices = np.where(t_test != predictions)[0]

    if len(error_indices) == 0:
        print("No prediction errors found!")
        return

    print(f"\nFound {len(error_indices)} prediction errors out of {len(t_test)} samples")
    print(f"Error rate: {len(error_indices)/len(t_test)*100:.2f}%")

    # 按置信度排序错误样本（置信度高但预测错误的样本更有趣）
    error_confidences = [probabilities[i][predictions[i]] for i in error_indices]
    sorted_indices = np.argsort(error_confidences)[::-1]  # 降序排列

    # 显示最有信心但预测错误的样本
    num_show = min(num_errors, len(error_indices))
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_show):
        idx = error_indices[sorted_indices[i]]

        # 显示图像
        axes[i].imshow(x_test[idx].reshape(28, 28), cmap='gray')

        # 设置标题
        true_label = t_test[idx]
        pred_label = predictions[idx]
        confidence = probabilities[idx][pred_label]

        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}',
                         color='red', fontsize=10)
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(num_show, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Most Confident Wrong Predictions', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(current_dir, 'error_analysis_pytorch.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Error analysis saved as '{save_path}'")


def main():
    """主函数"""
    print("PyTorch DeepConvNet Testing and Evaluation")
    print("=" * 50)

    # 检测设备
    device = get_device()

    # 检查模型文件是否存在
    model_path = os.path.join(current_dir, "deep_convnet_pytorch.pth")
    if not os.path.exists(model_path):
        print(f"\n错误: 模型文件 '{model_path}' 不存在!")
        print("请先运行以下命令之一来训练模型:")
        print("  python train_deepnet_pytorch.py  # 完整训练")
        print("  python demo_pytorch.py          # 快速演示训练")
        return

    # 加载MNIST测试数据
    print("\nLoading MNIST test dataset...")
    (_, _), (x_test, t_test) = load_mnist(flatten=False)

    # 可以选择使用完整测试集或减少数据量进行快速测试
    print(f"使用完整测试集: {len(x_test)} 个样本")
    # 如果想要快速测试，可以取消下面这行的注释
    # x_test, t_test = x_test[:1000], t_test[:1000]  # 快速测试

    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {t_test.shape}")

    # 加载训练好的模型
    print(f"\nLoading trained model from '{model_path}'...")
    model = load_trained_model(model_path, device)

    # 评估模型
    print("\nEvaluating model performance...")
    predictions, probabilities, accuracy = evaluate_model(model, x_test, t_test, device)

    print(f"\n Model Performance:")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 详细分类报告
    print("\nClassification Report:")
    print(classification_report(t_test, predictions, target_names=[str(i) for i in range(10)]))

    # 可视化样本预测
    print("\nGenerating sample predictions visualization...")
    plot_sample_predictions(x_test, t_test, predictions, probabilities, current_dir)

    # 绘制混淆矩阵
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(t_test, predictions, current_dir)

    # 错误分析
    print("\nAnalyzing prediction errors...")
    analyze_errors(x_test, t_test, predictions, probabilities, current_dir)

    print("\nEvaluation completed successfully!")
    print("\n生成的文件:")
    print("  - sample_predictions_pytorch.png  # 样本预测结果")
    print("  - confusion_matrix_pytorch.png    # 混淆矩阵")
    print("  - error_analysis_pytorch.png      # 错误分析")


if __name__ == "__main__":
    main()
