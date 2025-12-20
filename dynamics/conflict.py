import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from utils.sparse_ops import compute_sparse_aligned_cosine


class GradientConflictEngine(nn.Module):
    def __init__(self, topology_layer, backbone_layer):
        super().__init__()
        self.topology = topology_layer
        self.backbone = backbone_layer

    def forward(self, active_experts: list) -> torch.Tensor:
        """
        计算基于物理重叠的冲突演化损失。
        L_evolution = Sum_{i,j} [ Alpha_{ij} * ReLU( -Cosine(g_i, g_j) ) ]
        """
        grads = self.backbone.expert_grads
        if not grads or len(active_experts) < 2:
            return torch.tensor(0.0, device=self.topology.alpha.device)

        conflict_loss = 0.0
        overlap_count_log = 0  # 仅用于调试或监控

        # 遍历专家对
        for i_idx in range(len(active_experts)):
            for j_idx in range(len(active_experts)):
                expert_i = active_experts[i_idx]
                expert_j = active_experts[j_idx]

                if expert_i == expert_j:
                    continue

                # 1. 获取连接强度 (Alpha)
                # 我们只关心 expert_i 是否引用了 expert_j
                # 如果 Alpha[i, j] 很小，即便有物理冲突，系统也不应该负责（因为并非该连接导致的）
                connection_strength = self.topology.alpha[expert_i, expert_j]

                # 性能优化：如果根本没有连接意图，跳过昂贵的对齐计算
                if connection_strength < 1e-3:
                    continue

                # 2. 检查数据存在性
                if expert_i not in grads or expert_j not in grads:
                    continue

                # 3. 提取数据
                # grad_pkg = (gradient_tensor, indices_tensor)
                grad_i, idx_i = grads[expert_i]['u']
                grad_j, idx_j = grads[expert_j]['u']

                # 4. 执行稀疏对齐余弦计算 (Sparse Alignment)
                # 这一步精准地计算了物理层面的冲突
                cosine = compute_sparse_aligned_cosine(grad_i, idx_i, grad_j, idx_j)

                # 5. 冲突判定逻辑
                # cosine < 0 表示方向相反，即冲突。
                # 我们希望最小化 alpha * conflict。
                # conflict magnitude = ReLU(-cosine)
                conflict_magnitude = torch.relu(-cosine)

                # 6. 累加损失
                # 物理意义：如果我对你依赖很重 (Alpha高)，且我们在重叠区域打架 (Conflict高)，
                # 那么必须通过梯度下降减小 Alpha，即"断交"。
                conflict_loss += connection_strength * conflict_magnitude

        return conflict_loss