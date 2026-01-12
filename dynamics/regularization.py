import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn


class SystemRegularizer(nn.Module):
    """
    系统状态正则化器 (System State Regularizer)

    职责：
    1. 代谢惩罚 (Metabolic Cost): 模拟物理系统的能耗，抑制过多的连接，同时也要求其logit不会到负无穷以至于过早断开连接，即维持一定温度。
       L_metabolic = ||Alpha||_1
    2. 确定性惩罚 (Determinism Penalty): 迫使拓扑结构从模糊(Fuzzy)走向清晰(Binary)。
       通过最小化 Alpha 的熵或使用 Binary Cross Entropy 变体实现。

    物理意义：
    - 代谢惩罚防止热力学无限膨胀与无限断开。
    - 确定性惩罚模拟相变过程（从液态的各向同性变为固态的晶体结构）。
    """

    def __init__(self, topology_layer, lambda_metabolic=1e-4, lambda_determinism=1e-3):
        super().__init__()
        self.topology = topology_layer
        self.lambda_metabolic = lambda_metabolic
        self.lambda_determinism = lambda_determinism

    def forward(self) -> torch.Tensor:
        alpha = self.topology.alpha

        # 1. 代谢惩罚 (L1 Norm)
        # 稀疏诱导范数，迫使非必要的连接权重趋近于 0
        metabolic_loss = torch.mean(torch.abs(alpha))

        # 2. 确定性惩罚 (Determinism / Entropy Minimization)
        # 我们希望 alpha_ij 要么接近 0，要么接近 1
        # f(x) = x * (1 - x) 在 x=0.5 时最大，在 0/1 时最小
        # 注意：先将 alpha 限制在 [0, 1] 区间视为概率（通常 Alpha 会经过 Sigmoid 或保持正值）
        # 这里假设 Alpha 还是原始 Logits 或无约束权重，先做归一化模拟
        prob_alpha = torch.sigmoid(alpha)
        determinism_loss = torch.mean(prob_alpha * (1 - prob_alpha))

        total_reg_loss = (self.lambda_metabolic * metabolic_loss +
                          self.lambda_determinism * determinism_loss)

        return total_reg_loss
