import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List


def plot_classic_curves(history: Dict[str, List[float]], exp_name: str = "default"):
    """
    绘制经典的训练指标下降/上升曲线。

    Args:
        history: 包含 'step', 'total_loss', 'main_loss', 'conflict_loss', 'acc' 等键的字典
        exp_name: 实验名称，用于确定子文件夹路径
    """
    # 1. 路径锚定：确保保存到 analysis/pic/{exp_name}/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "pic", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 设置 Seaborn 风格，论文级绘图样式
    sns.set_theme(style="whitegrid")

    steps = history.get('step', range(len(history['total_loss'])))

    # ==========================
    # 图表 1: 损失函数分解 (Loss Decomposition)
    # ==========================
    plt.figure(figsize=(10, 6))

    # 绘制 Total Loss
    if 'total_loss' in history:
        sns.lineplot(x=steps, y=history['total_loss'], label='Total Loss', color='black', linewidth=2)

    # 绘制 Main Task Loss (通常是 CrossEntropy)
    if 'main_loss' in history:
        sns.lineplot(x=steps, y=history['main_loss'], label='Task Loss (CE)', color='blue', alpha=0.6, linestyle='--')

    # 绘制 Conflict Loss (通常数值较小，建议看情况是否用双轴，这里先画在一起)
    if 'conflict_loss' in history:
        sns.lineplot(x=steps, y=history['conflict_loss'], label='Conflict Penalty', color='red', alpha=0.8)

    # 绘制 Regularization Loss
    if 'reg_loss' in history:
        sns.lineplot(x=steps, y=history['reg_loss'], label='Reg Loss', color='green', alpha=0.5, linestyle=':')

    plt.title(f"Training Loss Curves - {exp_name}", fontsize=14)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.legend()
    plt.tight_layout()

    save_path_loss = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path_loss, dpi=300)
    plt.close()
    print(f"[Plotter] Loss curve saved to: {save_path_loss}")

    # ==========================
    # 图表 2: 准确率/稀疏度 (Metrics)
    # ==========================
    # 如果有准确率或稀疏度数据，额外画一张图
    metrics_to_plot = {}
    if 'task_acc' in history: metrics_to_plot['Task Accuracy'] = history['task_acc']
    if 'alpha_sparsity' in history: metrics_to_plot['Alpha Sparsity'] = history['alpha_sparsity']

    if metrics_to_plot:
        plt.figure(figsize=(10, 6))
        for name, values in metrics_to_plot.items():
            sns.lineplot(x=steps, y=values, label=name, linewidth=2)

        plt.title(f"System Metrics - {exp_name}", fontsize=14)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Value (0-1)", fontsize=12)
        plt.ylim(-0.05, 1.05)  # 通常这些指标在 0~1 之间
        plt.legend()
        plt.tight_layout()

        save_path_metrics = os.path.join(save_dir, "metrics_curve.png")
        plt.savefig(save_path_metrics, dpi=300)
        plt.close()
        print(f"[Plotter] Metrics curve saved to: {save_path_metrics}")