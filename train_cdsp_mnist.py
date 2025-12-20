import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# === 项目内引用 ===
from models.model import CDSPConfig, CDSPModel
from data.mixed_tasks import get_mixed_task_loaders
from analysis.visualizer import SystemMonitor
from analysis.plot_loss import plot_classic_curves


# === 配置类 (内部定义) ===
class ExperimentConfig:
    exp_name = "cdsp_mnist_3task"

    # 训练参数
    epochs = 10
    batch_size = 64
    learning_rate = 0.005

    # 模型参数 (Micro-Scale for Demo)
    image_size = 28
    patch_size = 4
    # 序列长度 = (28/4)^2 = 49
    seq_len = (image_size // patch_size) ** 2
    input_dim = patch_size * patch_size  # 4*4 = 16

    d_model = 64
    d_base = 256  # 物理基座大小 (4x d_model)
    num_experts = 8  # 8个专家
    num_tasks = 3  # 0: Digit, 1: Parity, 2: Magnitude

    # 词表大小设为10 (对应MNIST最大类别数)
    vocab_size = 10

    n_layers = 2
    n_heads = 4
    moe_top_k = 2


# === 视觉适配器 ===
def patchify_images(images, patch_size=4):
    """
    将图片 [B, 1, 28, 28] 转换为序列 [B, 49, 16]
    """
    B, C, H, W = images.shape
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
    patches = patches.transpose(1, 2)
    return patches


def adapt_model_for_vision(model, input_dim):
    """
    将 Transformer 的 Embedding 换成 Linear 投影
    """
    model.token_emb = nn.Linear(input_dim, model.config.d_model)
    nn.init.xavier_normal_(model.token_emb.weight)
    return model


# === 训练流程 ===
def train_cdsp():
    cfg = ExperimentConfig()  # 直接实例化内部配置类
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training CDSP-MoE on {device} ===")
    print(f"Experiment: {cfg.exp_name}")

    # 1. 准备数据
    train_loader, test_loader = get_mixed_task_loaders(batch_size=cfg.batch_size)

    # 2. 初始化模型
    model_config = CDSPConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_seq_len=cfg.seq_len,
        d_base=cfg.d_base,
        num_experts=cfg.num_experts,
        num_tasks=cfg.num_tasks,
        d_task_embed=16,
        moe_top_k=cfg.moe_top_k
    )
    model = CDSPModel(model_config)

    # 视觉适配
    model = adapt_model_for_vision(model, cfg.input_dim)
    model.to(device)

    # 3. 优化器 & 监视器 (Alpha 独立策略)
    alpha_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'alpha' in name:
            alpha_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': other_params, 'lr': cfg.learning_rate, 'weight_decay': 0.01},
        # Alpha: 高学习率，无权重衰减，完全由 Reward 和 Conflict 驱动
        {'params': alpha_params, 'lr': 0.05, 'weight_decay': 0.0}
    ])

    monitor = SystemMonitor(exp_name=cfg.exp_name)

    # 4. 训练循环
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()

        # [Tasks, Experts] 统计容器
        epoch_routing_stats = np.zeros((cfg.num_tasks, cfg.num_experts))

        epoch_loss = 0
        epoch_acc = 0
        samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

        for images, task_ids, labels in pbar:
            images, task_ids, labels = images.to(device), task_ids.to(device), labels.to(device)
            batch_size = images.size(0)

            # A. 数据转换
            inputs = patchify_images(images, cfg.patch_size)

            # B. 前向传播
            logits_seq, aux_loss, snapshot = model(inputs, task_ids)

            # C. 统计路由 (Routing Stats)
            with torch.no_grad():
                indices = snapshot['indices'].view(-1).cpu().numpy()
                B, S, K = snapshot['indices'].shape
                expanded_tasks = task_ids.view(B, 1, 1).expand(B, S, K).contiguous().view(-1).cpu().numpy()
                np.add.at(epoch_routing_stats, (expanded_tasks, indices), 1)

            # ---------------------------
            # D. 梯度清零 (Lagged Evolution 关键位置)
            # 在 Forward 后清零，确保 Conflict Engine 读到了上一轮梯度
            optimizer.zero_grad()
            # ---------------------------

            # E. Loss 计算
            logits = logits_seq.mean(dim=1)
            total_main_loss = 0
            correct = 0
            acc_breakdown = {}

            for t_id in range(cfg.num_tasks):
                mask = (task_ids == t_id)
                if not mask.any(): continue

                t_logits = logits[mask]
                t_labels = labels[mask]

                # [Masked Cross Entropy]
                # 因为数据集已经做了 0/1 映射，这里不需要切片
                loss_t = nn.CrossEntropyLoss()(t_logits, t_labels)
                total_main_loss += loss_t * mask.sum() / batch_size

                preds = t_logits.argmax(dim=-1)
                t_correct = (preds == t_labels).sum().item()
                correct += t_correct
                acc_breakdown[f"T{t_id}"] = t_correct / mask.sum().item()

            # 演化动力学 Loss (Conflict * 10.0 + L1)
            # Conflict Weight 保持 10.0
            conflict_weight = 10.0
            total_loss = total_main_loss + conflict_weight * aux_loss

            # F. 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # G. 记录
            epoch_loss += total_loss.item() * batch_size
            epoch_acc += correct
            samples += batch_size
            global_step += 1

            # 获取 Alpha 的稀疏度 (连接概率 < 0.1 的比例)
            alpha_prob = torch.sigmoid(model.layers[0].moe.topology.alpha)
            sparsity = (alpha_prob < 0.1).float().mean().item()

            monitor.log_step(global_step, {
                'total_loss': total_loss.item(),
                'main_loss': total_main_loss.item(),
                'conflict_loss': aux_loss.item(),
                'task_acc': correct / batch_size,
                'alpha_sparsity': sparsity
            })

            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Aux': f"{aux_loss.item():.3e}",
                'Acc': f"{correct / batch_size:.2%}",
                'T0': f"{acc_breakdown.get('T0', 0):.0%}",
                'T1': f"{acc_breakdown.get('T1', 0):.0%}",
                'T2': f"{acc_breakdown.get('T2', 0):.0%}"
            })

        # Epoch 结束统计
        avg_loss = epoch_loss / samples
        avg_acc = epoch_acc / samples
        print(f"Epoch {epoch + 1} Summary: Loss={avg_loss:.4f}, Acc={avg_acc:.2%}")

        # === 捕获演化快照 & 绘图 ===
        # 1. 捕获 Alpha (政治关系) 和 Routing (任务关注度)
        monitor.capture_snapshot(
            step=epoch,
            alpha_matrix=model.layers[0].moe.topology.alpha,
            routing_stats=epoch_routing_stats
        )

        # 2. 画图 (Alpha 会在内部自动 Sigmoid)
        monitor.plot_current_topology(step=epoch, filename=f"alpha_epoch_{epoch}.png")
        monitor.plot_current_routing(step=epoch, filename=f"task_routing_epoch_{epoch}.png")

    # 5. 训练结束分析
    print("=== Training Finished. Generating Final Reports... ===")

    # 生成最终演化图
    monitor.plot_topology_evolution(filename="final_alpha_evolution.png")
    monitor.plot_routing_evolution(filename="final_task_routing_evolution.png")
    plot_classic_curves(monitor.history, exp_name=cfg.exp_name)

    print(f"Results saved to: {monitor.save_dir}")


if __name__ == "__main__":
    train_cdsp()