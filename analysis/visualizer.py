import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional


class SystemMonitor:
    """
    CDSP 系统演化监视器 (重构版)
    严格分离 Alpha (专家关系) 和 Routing (任务-专家关系) 的可视化逻辑
    """

    def __init__(self, exp_name: str = "default"):
        # 路径设置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(current_dir, "pic", exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"[SystemMonitor] Artifacts will be saved to: {self.save_dir}")

        # 数据容器
        self.history = {
            'step': [], 'total_loss': [], 'main_loss': [],
            'conflict_loss': [], 'task_acc': [], 'alpha_sparsity': []
        }
        self.alpha_snapshots = {}  # 存 Alpha 矩阵 (N x N)
        self.routing_snapshots = {}  # 存 Task Routing 矩阵 (T x N)

    def log_step(self, step: int, metrics: Dict[str, float]):
        self.history['step'].append(step)
        for k, v in metrics.items():
            if k not in self.history: self.history[k] = []
            self.history[k].append(v)

    def capture_snapshot(self, step: int, alpha_matrix: torch.Tensor = None, routing_stats: np.ndarray = None):
        """
        捕获快照
        alpha_matrix: [Num_Experts, Num_Experts]
        routing_stats: [Num_Tasks, Num_Experts]
        """
        if alpha_matrix is not None:
            # 存原始数据，绘图时再处理
            self.alpha_snapshots[step] = alpha_matrix.detach().cpu().numpy()

        if routing_stats is not None:
            # routing_stats 应该是 [3, 8] (Tasks x Experts)
            self.routing_snapshots[step] = routing_stats.copy()

    # ====================================================
    # 1. 绘制 Alpha 矩阵 (内政: 专家 vs 专家)
    # ====================================================
    def plot_current_topology(self, step: int, filename=None):
        if step not in self.alpha_snapshots: return

        # 1. 获取原始 Logits
        raw_matrix = self.alpha_snapshots[step]  # [N, N]

        # 2. 关键: Sigmoid 变换 (Logits -> Probability)
        # 这样 -1.0 就会变成 0.26 (淡蓝)，1.0 变成 0.73 (红)
        matrix = 1 / (1 + np.exp(-raw_matrix))

        N = matrix.shape[0]

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(matrix, cmap="vlag", center=0.5, cbar=True, square=True,
                         vmin=0.0, vmax=1.0,  # 固定范围 0~1
                         annot=True, fmt=".2f", annot_kws={"size": 8},
                         xticklabels=[f"E{i}" for i in range(N)],
                         yticklabels=[f"E{i}" for i in range(N)])

        ax.set_title(f"Expert-to-Expert Political Alliances (Prob) at Epoch {step}")
        ax.set_xlabel("Source Expert (Resource Provider)")
        ax.set_ylabel("Target Expert (User)")

        plt.tight_layout()
        if filename is None: filename = f"alpha_epoch_{step}.png"
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    # ====================================================
    # 2. 绘制 Routing 矩阵 (外交: 任务 vs 专家)
    # ====================================================
    def plot_current_routing(self, step: int, filename=None):
        if step not in self.routing_snapshots: return

        # matrix shape: [Num_Tasks, Num_Experts] -> 应该是 [3, 8]
        stats = self.routing_snapshots[step]

        # 归一化: 每个任务的 Token 总数可能不同，按行归一化看百分比
        # row_sums: [Num_Tasks, 1]
        row_sums = stats.sum(axis=1, keepdims=True) + 1e-8
        norm_matrix = stats / row_sums

        num_tasks, num_experts = norm_matrix.shape

        plt.figure(figsize=(10, 5))  # 扁一点，因为是 3x8

        # 使用 Blues 配色，颜色越深代表选得越多
        ax = sns.heatmap(norm_matrix, cmap="Blues", cbar=True, square=False,  # 不是方阵
                         vmin=0.0, vmax=1.0,
                         annot=True, fmt=".2f", annot_kws={"size": 9},
                         xticklabels=[f"E{i}" for i in range(num_experts)],
                         yticklabels=[f"Task {i}" for i in range(num_tasks)])  # 强制显示 Task 标签

        ax.set_title(f"Task-to-Expert Attention Map at Epoch {step}")
        ax.set_xlabel("Experts")
        ax.set_ylabel("Tasks")
        plt.yticks(rotation=0)  # 让 Task 标签横着放

        plt.tight_layout()
        if filename is None: filename = f"task_routing_epoch_{step}.png"
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    # ====================================================
    # 3. 演化过程图 (Evolution)
    # ====================================================
    def plot_topology_evolution(self, filename="final_alpha_evolution.png"):
        """Alpha 演化拼图"""
        self._plot_evolution_generic(
            self.alpha_snapshots,
            title="Political Evolution (Expert Alliances)",
            xlabel="Source", ylabel="Target",
            filename=filename,
            is_alpha=True
        )

    def plot_routing_evolution(self, filename="final_routing_evolution.png"):
        """Routing 演化拼图"""
        self._plot_evolution_generic(
            self.routing_snapshots,
            title="Specialization Evolution (Task Assignments)",
            xlabel="Experts", ylabel="Tasks",
            filename=filename,
            is_alpha=False
        )

    def _plot_evolution_generic(self, data_dict, title, xlabel, ylabel, filename, is_alpha=True):
        steps = sorted(list(data_dict.keys()))
        if not steps: return

        # 选3个时间点: 开始、中间、结束
        if len(steps) < 3:
            selected = steps
        else:
            selected = [steps[0], steps[len(steps) // 2], steps[-1]]

        n = len(selected)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1: axes = [axes]

        for ax, step in zip(axes, selected):
            raw = data_dict[step]

            if is_alpha:
                # Alpha 处理: Sigmoid + Vlag Colormap + Square
                matrix = 1 / (1 + np.exp(-raw))
                sns.heatmap(matrix, ax=ax, cmap="vlag", center=0.5, cbar=False, square=True,
                            vmin=0.0, vmax=1.0, annot=False,
                            xticklabels=[f"E{i}" for i in range(matrix.shape[1])],
                            yticklabels=[f"E{i}" for i in range(matrix.shape[0])])
            else:
                # Routing 处理: Row Normalize + Blues Colormap + Rect
                row_sums = raw.sum(axis=1, keepdims=True) + 1e-8
                matrix = raw / row_sums
                sns.heatmap(matrix, ax=ax, cmap="Blues", cbar=False, square=False,
                            vmin=0.0, vmax=1.0, annot=True, fmt=".1f",
                            xticklabels=[f"E{i}" for i in range(matrix.shape[1])],
                            yticklabels=[f"T{i}" for i in range(matrix.shape[0])])

            ax.set_title(f"Epoch {step}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if not is_alpha: ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()