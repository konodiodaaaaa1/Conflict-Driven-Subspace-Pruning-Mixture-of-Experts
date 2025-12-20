import torch
import torch.nn as nn
import torch.nn.functional as F


class SubspaceTopology(nn.Module):
    """
    子空间拓扑控制器 (Subspace Topology Controller)

    修改日志：
    1. 初始化策略：采用 "Sculpting" (雕塑法)。
       - 初始非对角元素设为 0.1 (弱连接)，允许梯度立刻流动。
       - 初始对角元素设为 1.0 (强自指)，保证专家基本功能。
    2. 设备兼容性：修复了 get_subspace_indices 中的 device mismatch bug。
    """

    def __init__(self, num_experts: int, d_base: int, rank_quota: int):
        super().__init__()
        self.num_experts = num_experts
        self.d_base = d_base
        self.rank_quota = rank_quota

        # [Alpha]: N x N 矩阵 (专家间的结盟关系)
        # alpha[i, j] > 0 表示 Expert i 借用了 Expert j 的物理领地
        self.alpha = nn.Parameter(torch.empty(num_experts, num_experts))

        # [Pi]: N x D_base 矩阵 (固有的物理领地划分)
        self.register_buffer('pi', torch.zeros(num_experts, d_base))

        self._reset_parameters()

    def _reset_parameters(self):
        """
        初始化策略更新：
        我们不再从 0 开始学习连接，而是从“万物互联”开始，通过 Loss 剪枝。
        """
        # --- 1. 初始化 Pi (领地分配) ---
        # 保持不变：块状对角分布 (Hard Block Assignment)
        self.pi.zero_()
        block_size = self.d_base // self.num_experts
        for i in range(self.num_experts):
            start = i * block_size
            end = min((i + 1) * block_size, self.d_base)
            self.pi[i, start:end] = 1.0

        # --- 2. 初始化 Alpha (连接关系) [关键修改] ---

        # A. 全局弱连接 (Universal Weak Connection)
        # 将所有连接初始化为 0.1。
        # 意义：假设专家之间最初都有意愿合作，通过 Conflict Loss 把不需要的连接 "雕刻" (减) 掉。
        # 这样热力图一开始就是淡红色的，而不是惨白的。
        nn.init.constant_(self.alpha, 0.1)

        # B. 引入微小噪声 (Symmetry Breaking)
        # 加上 +/- 0.02 的噪声，防止所有专家梯度完全同步
        with torch.no_grad():
            noise = torch.randn_like(self.alpha) * 0.02
            self.alpha.add_(noise)

        # C. 强化对角线 (Self-Preservation)
        # 专家对自己领地的控制权必须是绝对的 (1.0)
        # 这样保证了即使没有合作，专家也能独立运作
        with torch.no_grad():
            self.alpha.fill_diagonal_(1.0)

        print(f"[Topology] Initialized Alpha: Diagonal=1.0, Off-diagonal~0.1")

    def get_influence_vector(self) -> torch.Tensor:
        """
        V = Sigmoid(Alpha) @ Pi
        [N, N] @ [N, D] -> [N, D]
        """
        connectivity = torch.sigmoid(self.alpha)
        influence_map = torch.matmul(connectivity, self.pi)
        return influence_map

    def get_subspace_indices(self, expert_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            expert_indices: [Batch] or [Active_Num]
        """
        # 自动对齐设备
        # 无论传入的 expert_indices 在 CPU 还是哪里，都强制移动到 Alpha 所在的 GPU
        expert_indices = expert_indices.to(self.alpha.device)

        # 1. 获取全量影响图 [N, D]
        influence_map = self.get_influence_vector()

        # 2. 提取目标专家的行 [Batch, D]
        active_influence = influence_map.index_select(dim=0, index=expert_indices)

        # 3. Top-K 截断
        values, indices = torch.topk(active_influence, k=self.rank_quota, dim=1)

        return indices, values

    def get_alpha_regularization(self) -> torch.Tensor:
        """
        返回 Alpha 矩阵用于计算 L1 正则项
        """
        return self.alpha
