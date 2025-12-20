import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PhysicalSubspaceBackbone(nn.Module):
    def __init__(self, d_model: int, d_base: int):
        super().__init__()
        self.d_model = d_model
        self.d_base = d_base
        self.U_base = nn.Parameter(torch.empty(d_model, d_base))
        self.V_base = nn.Parameter(torch.empty(d_base, d_model))
        self._reset_parameters()

        # 梯度暂存区：{expert_id: {'u': grad_tensor, 'v': grad_tensor}}
        self.expert_grads: Dict[int, Dict[str, torch.Tensor]] = {}
        # 钩子句柄列表，用于每轮清理
        self._hooks = []

    def _reset_parameters(self):
        nn.init.orthogonal_(self.U_base)
        nn.init.orthogonal_(self.V_base)

    def clear_gradient_buffer(self):
        """在每轮 Backward 前调用，清空上一轮的梯度记录"""
        self.expert_grads = {}
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _register_grad_hook(self, tensor: torch.Tensor, indices: torch.Tensor, expert_id: int, type_key: str):
        """
        注册钩子，同时捕获 梯度(Tensor) 和 物理索引(Indices)。
        """
        # 必须将 indices 从计算图中分离并转存，防止内存泄漏或计算图无限增长
        # indices 是整数索引，本身就不带梯度
        indices_captured = indices.detach().clone()

        def hook_fn(grad):
            if expert_id not in self.expert_grads:
                self.expert_grads[expert_id] = {}
            # 存储格式: (Gradient_Tensor, Physical_Indices_Tensor)
            self.expert_grads[expert_id][type_key] = (grad.detach().clone(), indices_captured)

        tensor.register_hook(hook_fn)

    def forward(self, x: torch.Tensor, subspace_indices: torch.Tensor, expert_id: Optional[int] = None) -> torch.Tensor:
        # 1. 动态子空间提取
        u_subset = self.U_base.index_select(dim=1, index=subspace_indices)
        v_subset = self.V_base.index_select(dim=0, index=subspace_indices)

        # 传递 subspace_indices 给 hook
        if expert_id is not None and self.training:
            self._register_grad_hook(u_subset, subspace_indices, expert_id, 'u')
            # v_subset 使用同样的 indices，但在 V_base 中代表行
            self._register_grad_hook(v_subset, subspace_indices, expert_id, 'v')

        # 2. 计算逻辑
        projected = torch.matmul(x, u_subset)
        projected = F.silu(projected)
        output = torch.matmul(projected, v_subset)

        return output