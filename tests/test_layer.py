import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import unittest
import torch.optim as optim

# Import from models/layer.py (assuming project root is in python path)
from models.layer import CDSPMoELayer


class TestCDSPMoELayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 设定微观环境参数
        self.d_model = 32
        self.d_base = 128  # 超完备基座
        self.num_experts = 4
        self.num_tasks = 10
        self.d_task_embed = 8

        self.layer = CDSPMoELayer(
            d_model=self.d_model,
            d_base=self.d_base,
            num_experts=self.num_experts,
            num_tasks=self.num_tasks,
            d_task_embed=self.d_task_embed,
            top_k=2
        )

    def test_dimensions_and_routing(self):
        """
        测试 1: 物理维度一致性与感知路由响应
        验证 Task ID 的注入是否真正改变了物理计算路径。
        """
        print("\n=== [Layer Test] Dimensions & Routing Sensitivity ===")
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, self.d_model)

        # Scenario A: Task 0
        task_a = torch.tensor([0, 0])
        out_a, _ = self.layer(x, task_a)

        # Scenario B: Task 1 (输入数据 x 相同，仅意图 Task ID 不同)
        task_b = torch.tensor([1, 1])
        out_b, _ = self.layer(x, task_b)

        # 1. 验证输出形状守恒
        self.assertEqual(out_a.shape, (batch_size, seq_len, self.d_model))

        # 2. 验证路由敏感性 (Routing Sensitivity)
        # 如果路由不仅看 x 还看 task_id，那么 out_a 和 out_b 应该不同
        diff = torch.norm(out_a - out_b)
        print(f"Output Difference between Task 0 & 1: {diff.item()}")
        self.assertGreater(diff.item(), 1e-5, "Layer failed to adapt to different Task IDs (Router is insensitive)")

    def test_evolution_dynamics_loop(self):
        """
        测试 2: 演化动力学闭环 (Forward -> Backward -> Memory -> Conflict)
        重点验证"滞后演化"机制是否生效。
        """
        print("\n=== [Layer Test] Lagged Evolution Dynamics ===")

        x = torch.randn(4, 8, self.d_model)  # Batch=4, Seq=8
        task_id = torch.tensor([0, 1, 0, 1])  # 混合任务，制造潜在冲突
        optimizer = optim.SGD(self.layer.parameters(), lr=0.1)

        # --- Step 1: 创世纪 (Genesis) ---
        # 此时没有梯度历史，Aux Loss 仅包含 Regularization
        optimizer.zero_grad()
        out1, aux_loss1 = self.layer(x, task_id)

        # 模拟主任务 Loss
        loss1 = out1.mean() + aux_loss1
        loss1.backward()

        # 验证：Backbone 是否记住了这次物理接触？
        # self.layer.backbone.expert_grads 应该非空
        grad_records = self.layer.backbone.expert_grads
        experts_active_step1 = list(grad_records.keys())
        print(f"Step 1 Active Experts (with grads recorded): {experts_active_step1}")
        self.assertGreater(len(experts_active_step1), 0, "Backbone failed to memorize gradients in Step 1")

        optimizer.step()  # 更新参数 (包括 Alpha)

        # --- Step 2: 演化 (Evolution) ---
        # 再次前向，此时 ConflictEngine 应该读取 Step 1 留下的记录
        optimizer.zero_grad()
        out2, aux_loss2 = self.layer(x, task_id)

        # 此时 aux_loss2 应该包含 conflict_loss
        print(f"Step 2 Aux Loss: {aux_loss2.item()}")

        # 验证 Loss 连通性：aux_loss2 必须能对 Topology.alpha 求导
        aux_loss2.backward()

        alpha_grad = self.layer.topology.alpha.grad
        grad_norm = torch.norm(alpha_grad)
        print(f"Alpha Gradient Norm after Step 2: {grad_norm.item()}")

        self.assertIsNotNone(alpha_grad, "Evolution logic is broken: Aux Loss did not reach Alpha parameters")
        # 注意：如果随即初始化的梯度碰巧没有负余弦冲突，grad可能为0，但graph必须连通
        # 在单元测试中，我们主要验证图的连通性(grad不是None)


if __name__ == '__main__':
    unittest.main()