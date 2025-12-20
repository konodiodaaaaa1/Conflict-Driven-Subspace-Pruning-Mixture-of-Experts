import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# === 复用项目模块 ===
from data.mixed_tasks import get_mixed_task_loaders
from analysis.plot_loss import plot_classic_curves
from analysis.visualizer import SystemMonitor


# === 1. 标准 MoE 层 ===
class StandardMoELayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k, num_tasks, d_task_embed):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.SiLU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])

        self.task_embedding = nn.Embedding(num_tasks, d_task_embed)
        self.gate = nn.Linear(d_model + d_task_embed, num_experts)

    def forward(self, x, task_id):
        batch_size, seq_len, _ = x.shape
        x_norm = F.layer_norm(x, x.shape[1:])
        t_emb = self.task_embedding(task_id).unsqueeze(1).expand(-1, seq_len, -1)

        gate_input = torch.cat([x_norm, t_emb], dim=-1)
        logits = self.gate(gate_input)

        topk_probs, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_probs = F.softmax(topk_probs, dim=-1)

        routing_snapshot = {
            'indices': topk_indices.detach(),
            'task_ids': task_id.detach()
        }

        flat_x = x.view(-1, x.shape[-1])
        final_output = torch.zeros_like(flat_x)
        expert_usage = torch.zeros(self.num_experts, device=x.device)

        for k in range(self.top_k):
            indices = topk_indices[:, :, k].view(-1)
            probs = topk_probs[:, :, k].view(-1, 1)
            for e_id in range(self.num_experts):
                mask = (indices == e_id)
                if mask.any():
                    expert_usage[e_id] += mask.sum()
                    expert_out = self.experts[e_id](flat_x[mask])
                    final_output[mask] += expert_out * probs[mask]

        final_output = final_output.view(batch_size, seq_len, -1)

        usage_mean = expert_usage.mean() + 1e-6
        usage_std = expert_usage.std()
        lb_loss = (usage_std / usage_mean) ** 2

        return final_output, lb_loss, routing_snapshot


class BaselineModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Linear(config.input_dim, config.d_model)
        nn.init.xavier_normal_(self.token_emb.weight)
        self.norm = nn.LayerNorm(config.d_model)

        self.layers = nn.ModuleList([
            StandardMoELayer(
                config.d_model, config.num_experts, config.moe_top_k,
                config.num_tasks, 16
            ) for _ in range(config.n_layers)
        ])

        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, task_id):
        h = self.token_emb(x)
        total_aux_loss = 0
        snapshots = []

        for layer in self.layers:
            h_res = h
            h_moe, aux, snap = layer(self.norm(h), task_id)
            h = h_res + h_moe
            total_aux_loss += aux
            snapshots.append(snap)

        logits = self.head(self.norm(h))
        logits = logits.mean(dim=1)

        return logits, total_aux_loss, snapshots[0]


# === 2. 配置与训练 ===
class BaselineConfig:
    exp_name = "baseline_mnist_3task"
    epochs = 10
    batch_size = 64
    learning_rate = 0.005
    patch_size = 4
    input_dim = 16
    d_model = 64
    num_experts = 8
    num_tasks = 3
    vocab_size = 10
    n_layers = 2
    moe_top_k = 2


def patchify_images(images, patch_size=4):
    B, C, H, W = images.shape
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
    patches = patches.transpose(1, 2)
    return patches


def train_baseline():
    cfg = BaselineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Baseline MoE on {device} ===")

    train_loader, test_loader = get_mixed_task_loaders(batch_size=cfg.batch_size)
    model = BaselineModel(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    monitor = SystemMonitor(exp_name=cfg.exp_name)

    global_step = 0

    # 这里的 dummy_alpha 仅用于占位，防止 plot_current_routing 内部逻辑报错
    # 但我们不会调用 plot_topology_evolution
    dummy_alpha_param = torch.zeros(cfg.num_experts, cfg.num_experts)

    for epoch in range(cfg.epochs):
        model.train()
        epoch_routing_stats = np.zeros((cfg.num_tasks, cfg.num_experts))

        epoch_loss = 0
        epoch_acc = 0
        samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for images, task_ids, labels in pbar:
            images, task_ids, labels = images.to(device), task_ids.to(device), labels.to(device)
            inputs = patchify_images(images, cfg.patch_size)

            logits, aux_loss, snapshot = model(inputs, task_ids)

            with torch.no_grad():
                B, S, K = snapshot['indices'].shape
                flat_indices = snapshot['indices'].view(-1).cpu().numpy()
                expanded_tasks = task_ids.view(B, 1, 1).expand(B, S, K).contiguous().view(-1).cpu().numpy()
                np.add.at(epoch_routing_stats, (expanded_tasks, flat_indices), 1)

            total_main_loss = 0
            correct = 0
            batch_size = images.size(0)
            acc_breakdown = {}

            for t_id in range(cfg.num_tasks):
                mask = (task_ids == t_id)
                if not mask.any(): continue

                t_logits = logits[mask]
                t_labels = labels[mask]

                loss_t = nn.CrossEntropyLoss()(t_logits, t_labels)
                total_main_loss += loss_t * mask.sum() / batch_size

                preds = t_logits.argmax(dim=-1)
                t_correct = (preds == t_labels).sum().item()
                correct += t_correct
                acc_breakdown[f"T{t_id}"] = t_correct / mask.sum().item()

            total_loss = total_main_loss + 0.01 * aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += total_loss.item() * batch_size
            epoch_acc += correct
            samples += batch_size

            # [关键修复] 手动填入 0，防止 plot_classic_curves 找不到 key 报错
            monitor.log_step(global_step, {
                'total_loss': total_loss.item(),
                'main_loss': total_main_loss.item(),
                'conflict_loss': 0.0,  # Baseline 无冲突，填0
                'alpha_sparsity': 0.0,  # Baseline 无Alpha，填0
                'task_acc': correct / batch_size
            })

            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Acc': f"{correct / batch_size:.2%}",
                'T0': f"{acc_breakdown.get('T0', 0):.0%}",
                'T1': f"{acc_breakdown.get('T1', 0):.0%}",
                'T2': f"{acc_breakdown.get('T2', 0):.0%}"
            })

        avg_loss = epoch_loss / samples
        avg_acc = epoch_acc / samples
        print(f"Epoch {epoch + 1} Summary: Loss={avg_loss:.4f}, Acc={avg_acc:.2%}")

        # 只捕获 routing stats，alpha 传 None
        monitor.capture_snapshot(
            epoch,
            alpha_matrix=None,
            routing_stats=epoch_routing_stats
        )

        # 只画 Task Routing 图
        monitor.plot_current_routing(step=epoch, filename=f"routing_epoch_{epoch}.png")

    print("=== Baseline Training Finished ===")

    # 画 Loss 曲线 (现在有了 conflict_loss=0 的数据，不会报错了)
    plot_classic_curves(monitor.history, exp_name=cfg.exp_name)

    # 画 Routing 演化图
    monitor.plot_routing_evolution(filename="final_routing_evolution.png")

    # [关键修复] 不再调用 plot_topology_evolution，因为没有 Alpha
    print(f"Results saved to: {monitor.save_dir}")


if __name__ == "__main__":
    train_baseline()