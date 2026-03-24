# coding: utf-8
"""
可视化工具模块
提供训练过程中损失函数和准确率的可视化功能
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_results(train_loss_list=None, train_acc_list=None, test_acc_list=None,
                         trainer=None, plot_type="standard",
                         loss_title="Loss", acc_title="Accuracy",
                         loss_xlabel="iteration", acc_xlabel="epochs",
                         title_prefix="", figsize=(12, 5),
                         save_path=None, save_dir="./plots/",
                         auto_save=False, filename_prefix="training_results",
                         show_plot=True):
    """
    统一的训练结果可视化函数，支持多种绘图模式和数据源

    参数:
        train_loss_list: 训练损失列表（当trainer为None时必需）
        train_acc_list: 训练准确率列表（当trainer为None时必需）
        test_acc_list: 测试准确率列表（可选）
        trainer: Trainer对象，包含train_loss_list, train_acc_list, test_acc_list（可选）
        plot_type: 绘图类型，可选值：
                  - "standard": 标准模式，完全自定义
                  - "simple": 简化模式，使用默认样式（适用于ch04/ch05）
                  - "cnn": CNN模式，支持标题前缀（适用于ch07）
        loss_title: 损失图标题
        acc_title: 准确率图标题
        loss_xlabel: 损失图x轴标签
        acc_xlabel: 准确率图x轴标签
        title_prefix: 标题前缀（在cnn模式下使用，如"LeNet", "AlexNet"等）
        figsize: 图形大小
        save_path: 完整保存路径（优先级最高）
        save_dir: 保存目录（当save_path为None且auto_save为True时使用）
        auto_save: 是否自动保存到save_dir目录
        filename_prefix: 自动保存时的文件名前缀
        show_plot: 是否显示图形

    返回:
        fig: matplotlib图形对象
        (ax1, ax2): 轴对象元组
    """
    import os

    # 数据源处理：优先使用trainer对象，否则使用传入的列表
    if trainer is not None:
        actual_train_loss_list = trainer.train_loss_list
        actual_train_acc_list = trainer.train_acc_list
        actual_test_acc_list = trainer.test_acc_list
    else:
        if train_loss_list is None or train_acc_list is None:
            raise ValueError("当trainer为None时，train_loss_list和train_acc_list不能为None")
        actual_train_loss_list = train_loss_list
        actual_train_acc_list = train_acc_list
        actual_test_acc_list = test_acc_list

    # 根据plot_type调整标题和标签
    if plot_type == "simple":
        actual_loss_title = "Loss"
        actual_acc_title = "Accuracy"
        actual_loss_xlabel = "iteration"
        actual_acc_xlabel = "epochs"
    elif plot_type == "cnn":
        # CNN模式：支持标题前缀
        actual_loss_title = f"{title_prefix} Training Loss" if title_prefix else "Training Loss"
        actual_acc_title = f"{title_prefix} Accuracy" if title_prefix else "Accuracy"
        actual_loss_xlabel = "iteration"
        actual_acc_xlabel = "epochs"
    else:  # plot_type == "standard"
        # 标准模式：使用传入的参数
        actual_loss_title = loss_title
        actual_acc_title = acc_title
        actual_loss_xlabel = loss_xlabel
        actual_acc_xlabel = acc_xlabel

    # 创建两个子图：一个显示损失函数推移，一个显示准确率推移
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 绘制损失函数推移图
    plot_loss_curve(ax1, actual_train_loss_list, title=actual_loss_title, xlabel=actual_loss_xlabel)

    # 绘制准确率推移图
    plot_accuracy_curve(ax2, actual_train_acc_list, actual_test_acc_list,
                       title=actual_acc_title, xlabel=actual_acc_xlabel)

    plt.tight_layout()

    # 处理保存路径
    final_save_path = None
    if save_path:
        # 优先使用指定的完整路径
        final_save_path = save_path
    elif auto_save:
        # 自动保存模式
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 根据plot_type和title_prefix生成文件名
        if plot_type == "cnn" and title_prefix:
            filename = f"{filename_prefix}_{title_prefix.lower()}.png"
        else:
            filename = f"{filename_prefix}_{plot_type}.png"

        final_save_path = os.path.join(save_dir, filename)

    # 保存图形（如果有保存路径）
    if final_save_path:
        # 确保保存目录存在
        save_directory = os.path.dirname(final_save_path)
        if save_directory and not os.path.exists(save_directory):
            os.makedirs(save_directory)

        plt.savefig(final_save_path, dpi=300, bbox_inches='tight')
        print(f"Training results plot saved as '{final_save_path}'")

    # 显示图形
    if show_plot:
        plt.show()

    return fig, (ax1, ax2)


def plot_loss_curve(ax, train_loss_list, title="Loss", xlabel="iteration", 
                   color='blue', label='train loss'):
    """
    绘制损失函数曲线
    
    参数:
        ax: matplotlib轴对象
        train_loss_list: 训练损失列表
        title: 图标题
        xlabel: x轴标签
        color: 线条颜色
        label: 图例标签
    """
    x_loss = np.arange(len(train_loss_list))
    ax.plot(x_loss, train_loss_list, label=label, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def plot_accuracy_curve(ax, train_acc_list, test_acc_list=None, title="Accuracy", 
                       xlabel="epochs", train_color='blue', test_color='red',
                       train_marker='o', test_marker='s', markevery=2):
    """
    绘制准确率曲线
    
    参数:
        ax: matplotlib轴对象
        train_acc_list: 训练准确率列表
        test_acc_list: 测试准确率列表（可选）
        title: 图标题
        xlabel: x轴标签
        train_color: 训练准确率线条颜色
        test_color: 测试准确率线条颜色
        train_marker: 训练准确率标记
        test_marker: 测试准确率标记
        markevery: 标记间隔
    """
    x_acc = np.arange(len(train_acc_list))
    
    # 绘制训练准确率
    if len(train_acc_list) > 10:  # 如果数据点较多，使用标记
        ax.plot(x_acc, train_acc_list, marker=train_marker, label='train acc',
                markevery=markevery, color=train_color, linestyle='-')
    else:  # 如果数据点较少，不使用标记
        ax.plot(x_acc, train_acc_list, label='train acc', color=train_color)
    
    # 绘制测试准确率（如果提供）
    if test_acc_list is not None:
        if len(test_acc_list) > 10:  # 如果数据点较多，使用标记
            ax.plot(x_acc, test_acc_list, marker=test_marker, label='test acc',
                    markevery=markevery, color=test_color, linestyle='--')
        else:  # 如果数据点较少，不使用标记
            ax.plot(x_acc, test_acc_list, label='test acc', color=test_color, linestyle='--')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True)


def setup_training_plot(figsize=(12, 5)):
    """
    设置训练图表的基本格式
    
    参数:
        figsize: 图形大小
        
    返回:
        fig: 图形对象
        (ax1, ax2): 轴对象元组
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    return fig, (ax1, ax2)


def plot_filters(filters, nx=8, title="Filters", show_plot=True):
    """
    可视化卷积滤波器

    参数:
        filters: 滤波器数组，形状为(filter_num, channels, height, width)
        nx: 每行显示的滤波器数量
        title: 图形标题
        show_plot: 是否显示图形

    返回:
        fig: matplotlib图形对象
    """
    FN = filters.shape[0]  # 滤波器数量
    ny = int(np.ceil(FN / nx))

    fig = plt.figure(figsize=(nx * 2, ny * 2))
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')

    if show_plot:
        plt.show()

    return fig


def visualize_conv_filters(network, layer_name='W1', nx=8, title_prefix="", show_plot=True):
    """
    可视化神经网络中的卷积滤波器

    参数:
        network: 神经网络对象，包含params字典
        layer_name: 要可视化的层的权重名称
        nx: 每行显示的滤波器数量
        title_prefix: 标题前缀
        show_plot: 是否显示图形

    返回:
        fig: matplotlib图形对象
    """
    if layer_name not in network.params:
        raise ValueError(f"Layer '{layer_name}' not found in network parameters")

    filters = network.params[layer_name]
    title = f"{title_prefix} {layer_name} Filters" if title_prefix else f"{layer_name} Filters"

    return plot_filters(filters, nx=nx, title=title, show_plot=show_plot)
