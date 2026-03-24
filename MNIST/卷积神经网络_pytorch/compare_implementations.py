# coding: utf-8
import sys
import os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# 导入NumPy实现
try:
    # 尝试从深度学习目录导入NumPy实现
    sys.path.append(os.path.join(project_root, 'MNIST', '卷积神经网络_深度学习'))
    from deep_convnet import DeepConvNet as DeepConvNetNumPy
    from common.trainer import Trainer
    numpy_available = True
except ImportError:
    print("Warning: NumPy implementation not available")
    numpy_available = False

# 导入PyTorch实现
from deep_convnet_pytorch import DeepConvNetPyTorch, get_device
from dataset.mnist import load_mnist


def benchmark_numpy_implementation(x_train, t_train, x_test, t_test, epochs=5):
    """测试NumPy实现的性能"""
    if not numpy_available:
        return None, None, None
    
    print("Benchmarking NumPy implementation...")
    start_time = time.time()
    
    # 创建NumPy网络
    network = DeepConvNetNumPy()
    
    # 创建训练器
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=epochs, mini_batch_size=100,
                      optimizer='Adam', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=1000, verbose=False)
    
    # 训练
    trainer.train()
    
    training_time = time.time() - start_time
    
    # 计算最终准确率
    final_accuracy = trainer.test_acc_list[-1] if trainer.test_acc_list else 0
    
    return training_time, final_accuracy, trainer.test_acc_list


def benchmark_pytorch_implementation(x_train, t_train, x_test, t_test, epochs=5, device='cpu'):
    """测试PyTorch实现的性能"""
    print(f"Benchmarking PyTorch implementation on {device}...")
    start_time = time.time()
    
    # 创建PyTorch网络
    model = DeepConvNetPyTorch()
    model = model.to(device)
    
    # 准备数据
    x_train_tensor = torch.FloatTensor(x_train).to(device)
    t_train_tensor = torch.LongTensor(t_train).to(device)
    x_test_tensor = torch.FloatTensor(x_test).to(device)
    t_test_tensor = torch.LongTensor(t_test).to(device)
    
    # 设置训练参数
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size = 100
    
    test_acc_list = []
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        
        # 训练一个epoch
        for i in range(0, len(x_train_tensor), batch_size):
            batch_x = x_train_tensor[i:i+batch_size]
            batch_t = t_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_t)
            loss.backward()
            optimizer.step()
        
        # 计算测试准确率
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == t_test_tensor).float().mean().item()
            test_acc_list.append(accuracy)
    
    training_time = time.time() - start_time
    final_accuracy = test_acc_list[-1] if test_acc_list else 0
    
    return training_time, final_accuracy, test_acc_list


def plot_comparison_results(numpy_results, pytorch_cpu_results, pytorch_gpu_results=None):
    """绘制比较结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 准备数据
    implementations = []
    times = []
    accuracies = []
    
    if numpy_results[0] is not None:
        implementations.append('NumPy')
        times.append(numpy_results[0])
        accuracies.append(numpy_results[1])
    
    if pytorch_cpu_results[0] is not None:
        implementations.append('PyTorch (CPU)')
        times.append(pytorch_cpu_results[0])
        accuracies.append(pytorch_cpu_results[1])
    
    if pytorch_gpu_results and pytorch_gpu_results[0] is not None:
        implementations.append('PyTorch (GPU)')
        times.append(pytorch_gpu_results[0])
        accuracies.append(pytorch_gpu_results[1])
    
    # 绘制训练时间比较
    colors = ['blue', 'green', 'red']
    bars1 = ax1.bar(implementations, times, color=colors[:len(implementations)])
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, time_val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    # 绘制最终准确率比较
    bars2 = ax2.bar(implementations, accuracies, color=colors[:len(implementations)])
    ax2.set_ylabel('Final Test Accuracy')
    ax2.set_title('Final Accuracy Comparison')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, acc_val in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(current_dir, 'implementation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison results saved as '{save_path}'")


def plot_accuracy_curves(numpy_results, pytorch_cpu_results, pytorch_gpu_results=None):
    """绘制准确率曲线比较"""
    plt.figure(figsize=(10, 6))

    epochs = range(1, 6)  # 5个epochs

    if numpy_results[2] is not None:
        plt.plot(epochs, numpy_results[2], 'b-o', label='NumPy', linewidth=2, markersize=6)

    if pytorch_cpu_results[2] is not None:
        plt.plot(epochs, pytorch_cpu_results[2], 'g-s', label='PyTorch (CPU)', linewidth=2, markersize=6)

    if pytorch_gpu_results and pytorch_gpu_results[2] is not None:
        plt.plot(epochs, pytorch_gpu_results[2], 'r-^', label='PyTorch (GPU)', linewidth=2, markersize=6)

    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Comparison Across Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.tight_layout()
    save_path = os.path.join(current_dir, 'accuracy_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Accuracy curves saved as '{save_path}'")


def main():
    """主函数"""
    print("Performance Comparison: NumPy vs PyTorch Implementation")
    print("=" * 60)
    
    # 加载数据
    print("Loading MNIST dataset...")
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    
    # 使用较小的数据集进行快速比较
    x_train, t_train = x_train[:6000], t_train[:6000]
    x_test, t_test = x_test[:1000], t_test[:1000]
    
    max_epochs=5

    print(f"Using {len(x_train)} training samples and {len(x_test)} test samples")
    print(f"Training for 5 epochs for fair comparison")
    print()
    
    # 检测可用设备
    device = get_device()
    has_gpu = torch.cuda.is_available()
    
    # 测试NumPy实现
    numpy_results = benchmark_numpy_implementation(x_train, t_train, x_test, t_test, epochs=max_epochs)
    
    # 测试PyTorch CPU实现
    pytorch_cpu_results = benchmark_pytorch_implementation(
        x_train, t_train, x_test, t_test, epochs=max_epochs, device='cpu'
    )
    
    # 测试PyTorch GPU实现（如果可用）
    pytorch_gpu_results = None
    if has_gpu:
        pytorch_gpu_results = benchmark_pytorch_implementation(
            x_train, t_train, x_test, t_test, epochs=max_epochs, device='cuda'
        )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    if numpy_results[0] is not None:
        print(f"NumPy Implementation:")
        print(f"  Training Time: {numpy_results[0]:.2f} seconds")
        print(f"  Final Accuracy: {numpy_results[1]:.4f}")
        print()
    
    if pytorch_cpu_results[0] is not None:
        print(f"PyTorch (CPU) Implementation:")
        print(f"  Training Time: {pytorch_cpu_results[0]:.2f} seconds")
        print(f"  Final Accuracy: {pytorch_cpu_results[1]:.4f}")
        if numpy_results[0] is not None:
            speedup = numpy_results[0] / pytorch_cpu_results[0]
            print(f"  Speedup vs NumPy: {speedup:.2f}x")
        print()
    
    if pytorch_gpu_results and pytorch_gpu_results[0] is not None:
        print(f"PyTorch (GPU) Implementation:")
        print(f"  Training Time: {pytorch_gpu_results[0]:.2f} seconds")
        print(f"  Final Accuracy: {pytorch_gpu_results[1]:.4f}")
        if numpy_results[0] is not None:
            speedup = numpy_results[0] / pytorch_gpu_results[0]
            print(f"  Speedup vs NumPy: {speedup:.2f}x")
        if pytorch_cpu_results[0] is not None:
            speedup = pytorch_cpu_results[0] / pytorch_gpu_results[0]
            print(f"  Speedup vs PyTorch CPU: {speedup:.2f}x")
        print()
    
    # 绘制比较图表
    print("Generating comparison plots...")
    plot_comparison_results(numpy_results, pytorch_cpu_results, pytorch_gpu_results)
    plot_accuracy_curves(numpy_results, pytorch_cpu_results, pytorch_gpu_results)
    
    print("\nComparison completed successfully!")


if __name__ == "__main__":
    main()
