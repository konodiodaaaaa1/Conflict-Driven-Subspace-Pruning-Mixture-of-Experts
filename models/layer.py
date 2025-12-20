import os

from torch import Tensor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from core.backbone import PhysicalSubspaceBackbone
from core.topology import SubspaceTopology
from dynamics.conflict import GradientConflictEngine
from dynamics.regularization import SystemRegularizer


class CDSPMoELayer(nn.Module):
    """
    CDSP-MoE 演化层 (Conflict-Driven Subspace Projection Layer)

    任务感知路由 (Task-Aware Routing)
    不再强制偏置，而是将 Task ID 作为特征注入 Router，让系统自发学习任务与专家(子空间)的映射关系。
    """

    def __init__(self,
                 d_model: int,
                 d_base: int,
                 num_experts: int,
                 num_tasks: int = 100,
                 d_task_embed: int = 32,
                 top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_task_embed = d_task_embed

        # --- 1. 物理层 (Physics) ---
        self.backbone = PhysicalSubspaceBackbone(d_model, d_base)

        # --- 2. 拓扑层 (Topology) ---
        # 秩配额 r = d_base / sqrt(N)
        rank_quota = int(d_base / (num_experts ** 0.5))
        self.topology = SubspaceTopology(num_experts, d_base, rank_quota)

        # --- 3. 感知路由层 (Perceptive Routing) ---
        # 任务嵌入层：将离散的 Task ID 映射为连续语义向量
        self.task_embedding = nn.Embedding(num_tasks, d_task_embed)

        # 路由器现在同时接收 [Token特征, 任务语义]
        # input_dim = d_model + d_task_embed
        self.router = nn.Linear(d_model + d_task_embed, num_experts)

        # --- 4. 动力学引擎 (Dynamics) ---
        self.conflict_engine = GradientConflictEngine(self.topology, self.backbone)
        self.regularizer = SystemRegularizer(self.topology)

    def _compute_router_logits(self, x: torch.Tensor, task_id: Optional[torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        计算路由概率 Logits。
        Logic: Concat(LayerNorm(x), TaskEmbed) -> Linear
        """
        batch_size, seq_len, _ = x.shape

        # 1. 特征状态归一化
        x_norm = F.layer_norm(x, x.shape[1:])  # [batch, seq, d_model]

        # 2. 任务状态注入
        if task_id is None:
            # 如果未提供 task_id，使用全 0 向量作为默认"无任务"状态
            # 这保证了接口的鲁棒性
            task_feat = torch.zeros(batch_size, seq_len, self.d_task_embed, device=x.device)
        else:
            # task_id: [batch]
            t_emb = self.task_embedding(task_id)  # [batch, d_task_embed]
            # 广播到序列长度: [batch, 1, d_task_embed] -> [batch, seq, d_task_embed]
            task_feat = t_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # 3. 状态融合 (State Fusion)
        # 将内容信息与意图信息在特征维度拼接
        router_input = torch.cat([x_norm, task_feat], dim=-1)  # [batch, seq, d_model + d_task]

        # 4. 决策
        logits = self.router(router_input)
        return logits

    def forward(self, x: torch.Tensor, task_id: Optional[torch.Tensor] = None) -> tuple[
        Tensor, Tensor | Any, dict[str, Any]]:

        batch_size, seq_len, _ = x.shape

        # 1. 路由决策
        router_logits = self._compute_router_logits(x, task_id)
        weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        routing_snapshot = {
            'indices': selected_experts.detach(),
            'weights': weights.detach()
        }

        # 2. 准备数据分发
        flat_x = x.view(-1, self.d_model)
        flat_experts = selected_experts.view(-1, self.top_k)
        flat_weights = weights.view(-1, self.top_k)

        final_output = torch.zeros_like(flat_x)
        active_experts_set = set()

        # [CRITICAL FIX]: 移除这里的 clear_gradient_buffer
        # if self.training:
        #     self.backbone.clear_gradient_buffer()  <-- 凶手就在这里，删掉它！

        # 3. 专家并行执行
        for k_idx in range(self.top_k):
            # ... (这部分代码保持不变) ...
            expert_ids = flat_experts[:, k_idx]
            expert_weights = flat_weights[:, k_idx].unsqueeze(1)
            unique_ids, _ = torch.sort(torch.unique(expert_ids))

            for expert_idx in unique_ids:
                e_id = expert_idx.item()
                active_experts_set.add(e_id)
                mask = (expert_ids == e_id)
                if not mask.any(): continue

                token_inputs = flat_x[mask]
                subspace_indices, subspace_strength = self.topology.get_subspace_indices(
                    torch.tensor([e_id], device=x.device))
                subspace_indices = subspace_indices[0]
                subspace_strength = subspace_strength[0]
                token_outputs = self.backbone(token_inputs, subspace_indices, expert_id=e_id)
                # 将计算结果乘以连接强度。
                # 逻辑：如果这个连接很弱 (strength小)，输出就弱。
                # 如果主任务发现这个输出很有用，梯度会告诉 strength 变大，
                # 从而带动 Alpha 变大
                gate_score = subspace_strength.mean().view(1, 1)
                token_outputs = token_outputs * gate_score
                weighted_out = token_outputs * expert_weights[mask]
                final_output[mask] += weighted_out

        final_output = final_output.view(batch_size, seq_len, self.d_model)

        # 4. 动力学演化 (Lagged Evolution)
        aux_loss = torch.tensor(0.0, device=x.device)

        if self.training:
            # 正则项
            reg_loss = self.regularizer()

            # 冲突项
            conflict_loss = torch.tensor(0.0, device=x.device)
            if len(active_experts_set) > 1:
                # 此时 Buffer 里还存着上一轮的梯度，正好拿来用！
                conflict_loss = self.conflict_engine(list(active_experts_set))

            aux_loss = reg_loss + conflict_loss

            # 用完之后再清理，为本轮 Backward 腾出空间
            # 注意：其实甚至不需要清理，因为 Hooks 通常是 overwrite 或者 accumuate
            # 但为了保险起见，我们在读取完之后清理它
            self.backbone.clear_gradient_buffer()

        return final_output, aux_loss, routing_snapshot
