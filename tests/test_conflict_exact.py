import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import unittest
from utils.sparse_ops import compute_sparse_aligned_cosine


class TestExactConflict(unittest.TestCase):
    def test_sparse_alignment_logic(self):
        print("\n=== Testing Sparse Alignment Math ===")

        d_model = 4
        rank = 3

        # 场景 1: 完全无重叠 (Physically Disjoint)
        # 专家 A 占领 [0, 1, 2], 专家 B 占领 [3, 4, 5]
        grad_a = torch.ones(d_model, rank)  # 全是 1
        idx_a = torch.tensor([0, 1, 2])

        grad_b = -torch.ones(d_model, rank)  # 全是 -1 (方向相反)
        idx_b = torch.tensor([3, 4, 5])

        # 虽然梯度数值看起来完全相反，但物理索引无重叠，冲突应为 0
        cos_1 = compute_sparse_aligned_cosine(grad_a, idx_a, grad_b, idx_b)
        print(f"Scenario 1 (No Overlap): Cosine = {cos_1.item()}")
        self.assertEqual(cos_1.item(), 0.0)

        # 场景 2: 部分重叠且冲突 (Partial Overlap & Conflict)
        # 专家 A: [0, 1, 2], 专家 B: [2, 3, 4] -> 重叠点是 2
        idx_a_2 = torch.tensor([0, 1, 2])
        idx_b_2 = torch.tensor([2, 3, 4])

        # 构造梯度：在非重叠区保持一致，在重叠区(index 2)设为反向
        grad_a_2 = torch.zeros(d_model, rank)
        grad_a_2[:, 2] = 1.0  # Index 2 处梯度为正

        grad_b_2 = torch.zeros(d_model, rank)
        grad_b_2[:, 0] = -1.0  # Index 2 (在 B 中是第 0 列) 处梯度为负

        cos_2 = compute_sparse_aligned_cosine(grad_a_2, idx_a_2, grad_b_2, idx_b_2)
        print(f"Scenario 2 (Overlap Index 2, Opposing Grads): Cosine = {cos_2.item()}")

        # 应该检测到 -1.0 的完全负相关
        self.assertAlmostEqual(cos_2.item(), -1.0, places=4)


if __name__ == '__main__':
    unittest.main()