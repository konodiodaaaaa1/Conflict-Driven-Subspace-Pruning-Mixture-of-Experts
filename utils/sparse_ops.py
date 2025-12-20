import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch


def compute_sparse_aligned_cosine(
        grad_a: torch.Tensor, indices_a: torch.Tensor,
        grad_b: torch.Tensor, indices_b: torch.Tensor
) -> torch.Tensor:
    """
    计算两个稀疏梯度在物理重叠区域的余弦相似度。

    Args:
        grad_a: [d_model, rank_quota] - 专家 A 的梯度张量
        indices_a: [rank_quota] - 专家 A 对应的物理基座索引
        grad_b: [d_model, rank_quota] - 专家 B 的梯度张量
        indices_b: [rank_quota] - 专家 B 对应的物理基座索引

    Returns:
        cosine_sim: 标量 Tensor。
            如果在物理上无重叠，返回 0.0。
            如果有重叠，仅计算重叠维度的梯度的余弦相似度。
    """
    # 1. 寻找物理交集 (Intersection of Physical Indices)
    # 使用 Python Set 操作在 CPU 上快速求交集（对于 rank_quota < 1000 这种规模非常快）
    set_a = set(indices_a.tolist())
    set_b = set(indices_b.tolist())
    common_indices = list(set_a.intersection(set_b))

    if not common_indices:
        return torch.tensor(0.0, device=grad_a.device)

    # 转换回 Tensor 以便进行索引操作
    common_tensor = torch.tensor(common_indices, device=grad_a.device, dtype=torch.long)

    # 2. 映射逻辑 (Mapping Logic)
    # 我们需要知道 common_indices 中的每个物理索引，分别对应 grad_a 和 grad_b 的第几列
    # 例如: indices_a = [1, 5, 9], common = [5] -> pos_a = [1]

    # 使用 torch.isin 或 mask 查找位置
    # 注意：这里假设 indices 中无重复元素 (Top-K 保证了这一点)
    mask_a = torch.isin(indices_a, common_tensor)
    mask_b = torch.isin(indices_b, common_tensor)

    # 3. 提取重叠子梯度 (Extract Overlapping Sub-gradients)
    # sub_grad_a shape: [d_model, num_common]
    # 我们需要确保提取出的列是按照 common_indices 的顺序对齐的
    # 由于 isin 不保证顺序，我们需要更严格的 gather 逻辑

    # 严密实现：构建从 common_val 到 local_index 的查找表不可导，
    # 但我们只需要取出数值。
    # 既然我们要计算的是点积，列的顺序不影响总和，只要 A 的列和 B 的列是对应同一个物理索引即可。
    # 为了简单且正确，我们遍历 common_indices (数量很少，通常 < 64)

    vecs_a = []
    vecs_b = []

    # 将 indices 转为字典加速查找: {physical_idx: local_col_idx}
    dict_a = {idx.item(): i for i, idx in enumerate(indices_a)}
    dict_b = {idx.item(): i for i, idx in enumerate(indices_b)}

    col_idx_a = [dict_a[c] for c in common_indices]
    col_idx_b = [dict_b[c] for c in common_indices]

    # 提取对应列
    # shape: [d_model, num_common]
    overlap_grad_a = grad_a[:, col_idx_a]
    overlap_grad_b = grad_b[:, col_idx_b]

    # 4. 计算余弦相似度 (Cosine Similarity)
    # 将矩阵展平为向量进行计算
    flat_a = overlap_grad_a.flatten()
    flat_b = overlap_grad_b.flatten()

    dot_product = torch.dot(flat_a, flat_b)
    norm_a = torch.norm(flat_a) + 1e-8
    norm_b = torch.norm(flat_b) + 1e-8

    return dot_product / (norm_a * norm_b)