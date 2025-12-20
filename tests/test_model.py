import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import unittest
import torch.optim as optim
from models.model import CDSPModel, CDSPConfig


class TestCDSPModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1024)
        # 配置一个小型但完整的模型进行测试
        self.config = CDSPConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,  # 2层堆叠，验证深层传递
            n_heads=4,
            n_kv_heads=2,  # GQA 测试: 4 Query Heads vs 2 KV Heads
            max_seq_len=50,
            d_base=128,  # 物理基座
            num_experts=4,
            num_tasks=5,
            d_task_embed=16,
            moe_top_k=2
        )
        self.model = CDSPModel(self.config)

    def test_gqa_rope_integration(self):
        """
        测试 1: GQA 注意力机与 RoPE 的几何正确性
        验证输入经过 Embedding -> RoPE -> GQA -> MoE 的完整通路形状。
        """
        print("\n=== [Model Test] GQA & RoPE Integration ===")
        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        task_id = torch.tensor([0, 1])  # Batch 中不同任务

        # Forward Pass
        logits, aux_loss = self.model(input_ids, task_id)

        # 1. 验证 Logits 形状 (Vocab 投影)
        expected_shape = (batch_size, seq_len, self.config.vocab_size)
        self.assertEqual(logits.shape, expected_shape)

        print(f"Logits Shape: {logits.shape} (Matches Expectation)")
        print(f"Total Aux Loss (Accumulated): {aux_loss.item()}")

    def test_end_to_end_training_step(self):
        """
        测试 2: 全网梯度流 (End-to-End Gradient Flow)
        验证从 LM Head 到 底层 Embeddings 以及 Topology Alpha 的梯度是否连通。
        """
        print("\n=== [Model Test] End-to-End Optimization ===")
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001)

        input_ids = torch.randint(0, self.config.vocab_size, (2, 10))
        task_id = torch.tensor([0, 0])
        labels = torch.randint(0, self.config.vocab_size, (2, 10))

        # --- Step 1 ---
        optimizer.zero_grad()
        logits, aux_loss = self.model(input_ids, task_id)

        # 计算交叉熵损失 (Target Task)
        loss_fn = torch.nn.CrossEntropyLoss()
        main_loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 总损失 = 任务损失 + 演化损失
        total_loss = main_loss + aux_loss
        total_loss.backward()

        # 检查梯度是否存在
        # 1. 检查底层 Embedding
        self.assertIsNotNone(self.model.token_emb.weight.grad, "Gradient failed to reach Token Embeddings")

        # 2. 检查某一层专家的 Alpha (拓扑结构)
        # 注意：Step 1 时 Conflict Loss 可能为0，但 Regularization Loss 总是存在的
        layer_0_alpha_grad = self.model.layers[0].moe.topology.alpha.grad
        self.assertIsNotNone(layer_0_alpha_grad, "Gradient failed to reach Topology Alpha")

        # 3. 检查注意力层
        attn_grad = self.model.layers[0].attention.wq.weight.grad
        self.assertIsNotNone(attn_grad, "Gradient failed to reach Attention weights")

        optimizer.step()
        print("Optimization Step 1 completed successfully.")

        # --- Step 2 (Check Lagged Conflict) ---
        # 再次运行以触发潜在的 Conflict Engine (因为 Step 1 留下了梯度记录)
        optimizer.zero_grad()
        logits_2, aux_loss_2 = self.model(input_ids, task_id)
        (logits_2.mean() + aux_loss_2).backward()

        print("Optimization Step 2 completed (Lagged Conflict Triggered).")


if __name__ == '__main__':
    unittest.main()