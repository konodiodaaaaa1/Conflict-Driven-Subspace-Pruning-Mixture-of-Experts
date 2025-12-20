import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from models.layer import CDSPMoELayer


# ==========================================
# 1. 基础组件 (Infrastructure Components)
# ==========================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    比 LayerNorm 去掉了均值中心化，计算更高效，且在深层网络中梯度更稳定。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [batch, seq, dim]
        # 计算 RMS: sqrt(mean(x^2))
        norm_x = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算 RoPE 的旋转频率矩阵 (Complex Exponential).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    应用旋转位置编码。
    将 Query 和 Key 视为复数向量，乘以旋转矩阵 freqs_cis。
    """
    # 重塑为复数形式: [batch, seq, head, dim/2] -> complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 广播维度对齐
    freqs_cis = freqs_cis[:xq.shape[1]].view(1, xq.shape[1], 1, -1)

    # 复数乘法 (旋转)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==========================================
# 2. 升级版注意力机 (The Attention Machine)
# ==========================================

class GroupedQueryAttention(nn.Module):
    """
    GQA (Grouped Query Attention) - 这里的"注意力机"。

    相比于 MHA (每个 Head 都有 KV) 和 MQA (所有 Head 共享 1 个 KV)，
    GQA 处于中间态：将 Query 分成 n_groups，每组共享一个 KV。
    这是目前性价比最高的注意力机制。
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # 重复倍数

        # 投影层
        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape

        # 1. 投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. 重塑 Heads
        # xq: [batch, seq, n_heads, d_head]
        xq = xq.view(batch_size, seq_len, self.n_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.d_head)

        # 3. 应用 RoPE (如果提供)
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 4. KV 重复 (GQA 核心逻辑)
        # 将 KV Heads 复制扩展以匹配 Query Heads 的数量
        if self.n_rep > 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, self.n_rep, 1).reshape(
                batch_size, seq_len, self.n_heads, self.d_head)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, self.n_rep, 1).reshape(
                batch_size, seq_len, self.n_heads, self.d_head)

        # 5. 转置以便计算 Attention
        # [batch, n_heads, seq, d_head]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 6. Scaled Dot-Product Attention
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores + mask  # mask 通常为 -inf

        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, xv)

        # 7. 输出重组
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


# ==========================================
# 3. 核心计算块 (The Stack Block)
# ==========================================

class CDSPBlock(nn.Module):
    """
    CDSP Transformer Block

    Structure:
    x -> RMSNorm -> GQA (Attention) -> Residual -> x
    x -> RMSNorm -> CDSP-MoE (Expert FFN) -> Residual -> x
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model

        # Attention Machine
        self.norm1 = RMSNorm(config.d_model)
        self.attention = GroupedQueryAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads
        )

        # Expert Machine (CDSP-MoE)
        self.norm2 = RMSNorm(config.d_model)
        self.moe = CDSPMoELayer(
            d_model=config.d_model,
            d_base=config.d_base,
            num_experts=config.num_experts,
            num_tasks=config.num_tasks,
            d_task_embed=config.d_task_embed,
            top_k=config.moe_top_k
        )

    def forward(self, x, freqs_cis, mask, task_id):
        # 1. Attention Path
        h = x + self.attention(self.norm1(x), freqs_cis, mask)

        # 2. MoE Path (Subspace Projection)
        # CDSP Layer 返回 (output, aux_loss, snapshot)
        moe_out, aux_loss, snapshot = self.moe(self.norm2(h), task_id)
        out = h + moe_out

        return out, aux_loss, snapshot


# ==========================================
# 4. 整体模型 (Total System)
# ==========================================

class CDSPModel(nn.Module):
    """
    CDSP-MoE 完整模型架构

    特点：
    - 纯 Decoder 架构 (GPT-style)
    - 集成 RoPE 旋转位置编码
    - 集成 GQA 注意力机制
    - 深度集成冲突驱动的 MoE 层
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token Embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Precompute RoPE frequencies (Cached)
        # 预计算足够长的序列长度，例如 4096
        self.freqs_cis = precompute_freqs_cis(
            config.d_model // config.n_heads,
            config.max_seq_len * 2
        )

        # Transformer Blocks
        self.layers = nn.ModuleList([
            CDSPBlock(config) for _ in range(config.n_layers)
        ])

        # Final Norm & Head
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying (可选，通常能提升效果)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, task_id: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: [batch, seq_len]
            task_id: [batch] (可选，用于 CDSP 路由学习)
        """
        batch_size, seq_len = input_ids.shape[:2]

        # 1. Embedding
        h = self.token_emb(input_ids)

        # 2. RoPE Preparation
        # 动态切片当前序列长度的频率矩阵
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)

        # 3. Causal Mask
        # 生成下三角掩码
        mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=1)

        # 4. Layer Stack Execution
        total_aux_loss = 0.0
        layer_snapshots = []

        for layer in self.layers:
            # 传入 task_id 供 CDSP 层使用
            h, layer_loss, snap = layer(h, freqs_cis, mask, task_id)
            total_aux_loss += layer_loss
            layer_snapshots.append(snap)

        # 5. Final Output
        h = self.norm_f(h)
        logits = self.lm_head(h)

        return logits, total_aux_loss, layer_snapshots[0]


# ==========================================
# 5. 配置类 (Configuration)
# ==========================================

class CDSPConfig:
    def __init__(self,
                 vocab_size=32000,
                 d_model=512,
                 n_layers=6,
                 n_heads=8,
                 n_kv_heads=2,  # GQA: KV Heads 少于 Query Heads
                 max_seq_len=1024,
                 d_base=1024,  # 物理基座超完备维度
                 num_experts=16,
                 num_tasks=100,
                 d_task_embed=32,
                 moe_top_k=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len

        self.d_base = d_base
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.d_task_embed = d_task_embed
        self.moe_top_k = moe_top_k
