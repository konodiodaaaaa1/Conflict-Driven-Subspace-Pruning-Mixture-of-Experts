# CDSP-MoE 技术架构白皮书 (Technical Architecture Document)

**项目名称**: Conflict-Driven Subspace Pruning Mixture-of-Experts (CDSP-MoE)

**核心理念**: 利用“滞后梯度”计算参数更新方向的冲突程度，将其作为结构演化的负反馈信号，替代传统 MoE 的负载均衡辅助损失，实现神经网络拓扑结构的自适应稀疏化与模块化。

[![arXiv](https://img.shields.io/badge/arXiv-2512.20291-b31b1b.svg)](https://arxiv.org/abs/2512.20291)

Paper: [Mixture-of-Experts with Gradient Conflict-Driven Subspace Topology Pruning for Emergent Modularity](https://arxiv.org/abs/2512.20291)

---

## 1. 物理层：共享子空间基座 (Physical Subspace Backbone)

不同于传统 MoE 中专家权重矩阵的物理隔离，CDSP 采用“超完备共享参数空间”设计，所有专家共享同一组底层参数，通过掩码进行逻辑区分。

* **定义**: 物理基座由两个正交初始化的超大参数矩阵组成：

$$
\mathbf{U}_{base} \in \mathbb{R}^{D_{model} \times D_{base}}, \quad \mathbf{V}_{base} \in \mathbb{R}^{D_{base} \times D_{model}}
$$

其中 $D_{base} \gg D_{model}$ (在实验设置中 $D_{base} = 4 \times D_{model}$)。

* **物理初始分区 ($\mathbf{\Pi}$)**:
    * 系统维护一个固定缓冲区 `self.pi` ($\mathbf{\Pi} \in \{0,1\}^{N \times D_{base}}$)，用于定义专家的初始参数索引范围。
    * **初始化策略**: 块状对角 (Block Diagonal) 划分。专家 $i$ 初始对应的参数区间为 $[i \times B, (i+1) \times B)$，其中 $B = D_{base}/N$。

---

## 2. 拓扑层：动态连接矩阵 (Topology Layer)

该层定义了逻辑专家与物理参数之间的连接权重，是一个可学习的加权有向图。

* **拓扑矩阵 ($\mathbf{A}$)**:
    * 参数 `self.alpha` ($\mathbf{A} \in \mathbb{R}^{N \times N}$)。
    * **数学定义**: $\mathbf{A}_{ij}$ 表示逻辑专家 $i$ 对 初始归属于 $j$ 的物理参数区间 $\mathbf{\Pi}_j$ 的**连接强度**。

* **结构初始化 (Structural Initialization)**:
    * 采用“通用弱连接”与“偏置初始化”策略，而非随机初始化，以确保训练初期的梯度流动。
    * **对角线 (Self-Loop)**: $\mathbf{A}_{ii} \leftarrow 1.0$ (保证专家自身的参数稳定性)。
    * **非对角线 (Cross-Link)**: $\mathbf{A}_{ij} \leftarrow 0.1 + \epsilon$ (引入微弱的全局连接与噪声，$\epsilon \sim \mathcal{N}(0, 0.01)$)。
    * **代码对应**: `topology.py` 中的 `_reset_parameters`。

* **参数控制力投影 (Control Projection)**:
    专家 $i$ 对物理维度 $k$ 的实际激活强度 $\mathbf{I}_i \in \mathbb{R}^{D_{base}}$ 计算如下：

$$
\mathbf{I}_i = \sigma(\mathbf{A}_{i, :}) \cdot \mathbf{\Pi}
$$

(即：专家 $i$ 对物理基座的控制力，是其对所有初始分区的连接权重 $\sigma(\mathbf{A})$ 与分区映射 $\mathbf{\Pi}$ 的线性组合)。

---

## 3. 感知路由层 (Perceptive Routing Layer)

路由层负责根据输入特征分配计算资源。为防止模型仅依赖任务 ID 进行简单查表（Shortcut Learning），引入了对抗性掩码机制。

* **输入归一化与融合 (Normalization & Fusion)**:
    为了消除不同 Token 模长对路由决策的干扰，Router 输入端的 Token 特征首先经过 LayerNorm 处理，再与任务嵌入拼接：

$$
h_{in} = [ \text{LayerNorm}(x) \oplus v_{task} ]
$$

* **对抗性任务掩码 (Adversarial Task Masking)**:
    引入随机掩码变量 $\mathcal{M}$。在训练过程中，以概率 $p_{drop}$ 将任务嵌入向量置零，强制 Router 挖掘输入内容 $x_{norm}$ 的内在特征。

$$
v_{task} = \mathcal{M} \cdot \text{Embed}(t)
$$

其中掩码 $\mathcal{M}$ 的生成逻辑为：

$$
\mathcal{M} = \begin{cases}
0 & \text{if } t \text{ is None or } \xi < p_{drop} \\
1 & \text{otherwise}
\end{cases}
$$

($\xi \sim U[0,1]$ 为随机均匀分布变量)。

* **门控决策 (Gating Decision)**:
    Router 基于融合特征计算专家的激活权重：

$$
G(x, t) = \text{Softmax}(W_g \cdot h_{in})
$$

* **代码对应**: `layer.py` 中的 `_compute_router_logits`。

---

## 4. 前向传播动力学 (Forward Dynamics)

当专家 $i$ 被激活时，通过动态索引机制调用物理基座的特定子集进行计算。

1. **子空间寻址 (Addressing)**:
    根据控制力 $\mathbf{I}_i$ 选取 Top-$r$ 个物理维度索引 $\mathcal{S}_i$ (秩配额 $r$)。
    * **注**: 采用 **平方根缩放 (Square Root Scaling)** 以维持稀疏度平衡。
    * **公式**:

$$
r = \lfloor \frac{D_{base}}{\sqrt{N}} \rfloor
$$

* **代码对应**: `topology.py` -> `get_subspace_indices`。

2. **稀疏计算 (Sparse Computation)**:
    仅提取 $\mathcal{S}_i$ 对应的参数切片进行矩阵乘法。

$$
y_{raw} = \text{SiLU}(x \cdot \mathbf{U}_{base}[\mathcal{S}_i]) \cdot \mathbf{V}_{base}[\mathcal{S}_i]^T
$$

* **代码对应**: `backbone.py` -> `forward` (使用 `index_select`)。

3. **强度调制与梯度桥接 (Strength Modulation & Gradient Bridge)**:
    为了建立从主任务损失 $\mathcal{L}_{task}$ 到拓扑矩阵 $\mathbf{A}$ 的可微路径，系统将子空间的平均激活强度作为门控乘数作用于输出：

$$
S_{gate, i} = \text{Mean}( \text{TopK}(\mathbf{I}_i) )
$$

$$
y_{final} = y_{raw} \cdot S_{gate, i} \cdot w_{router, i}
$$

* **物理意义**: 
    * **前向**: 即使 Router 选中了专家 $i$，如果拓扑层判定专家 $i$ 对当前子空间的控制力（Strength）很弱，其输出也会被抑制。
    * **反向**: $\mathcal{L}_{task}$ 的梯度会通过 `GateScore` 回传至 $\mathbf{I}_i$，进而更新 $\mathbf{A}$。这意味着如果某个子空间对降低任务 Loss 有帮助，系统会自动增强相关的连接强度。
* **代码对应**: `layer.py` 中的 `token_outputs * gate_score`。

---

## 5. 梯度冲突优化 (Gradient Conflict Optimization)

这是 CDSP 的核心结构优化机制。利用滞后梯度检测参数更新方向的冲突，作为结构剪枝的依据。

### 5.1 梯度采样 (Gradient Sampling)

利用 PyTorch Hook 机制，在反向传播阶段捕获每个专家在物理基座上的梯度投影。

* **数据结构**: `expert_grads[i] = (GradTensor, PhysicalIndices)`。
* **时序逻辑**: `optimizer.zero_grad()` 位于 `forward` 之后、`backward` 之前。因此，Forward 阶段读取的 `expert_grads` 为 **Step t-1** 的梯度信息。

### 5.2 空间冲突度量 (Spatial Conflict Metric)

对于当前 Batch 中同时激活的一对专家 $(i, j)$，计算其在**物理参数重叠区域**的梯度余弦相似度。

1.  **参数交集**: $\mathcal{K} = \mathcal{S}_i \cap \mathcal{S}_j$。若交集为空，冲突为 0。
2.  **梯度对齐**: 提取交集 $\mathcal{K}$ 对应的梯度切片 $g_i^{\mathcal{K}}, g_j^{\mathcal{K}}$。
3.  **余弦相似度计算**:

$$
\cos(i, j) = \frac{g_i^{\mathcal{K}} \cdot g_j^{\mathcal{K}}}{\|g_i^{\mathcal{K}}\| \|g_j^{\mathcal{K}}\| + \epsilon}
$$

4.  **干扰惩罚项**:

$$
\text{Conflict}_{ij} = \text{ReLU}(-\cos(i, j))
$$

* **物理意义**: 当且仅当梯度方向**相反**（负余弦，即参数更新方向冲突）时，产生正向惩罚。正交或同向被视为参数复用或协同，不产生惩罚。
* **代码对应**: `conflict.py` -> `compute_sparse_aligned_cosine`。

---

## 6. 全局结构演化目标 (Structural Evolution Objective)

总损失函数通过多目标优化，强制系统向“低熵、低干扰、模块化”的结构收敛。

$$
\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{conf} \mathcal{L}_{conflict} + \mathcal{L}_{reg}
$$

### 6.1 冲突驱动剪枝 (Conflict-Driven Pruning)

如果专家 $i$ 与专家 $j$ 存在高强度的连接，且二者在参数更新上存在严重冲突，则通过梯度下降抑制连接强度。

$$
\mathcal{L}_{conflict} = \sum_{i \neq j} \sigma(\mathbf{A}_{ij}) \cdot \text{Conflict}_{ij}
$$

* **权重**: $\lambda_{conf} = 10.0$ (强约束)。
* **注**: 代码中显式使用 `sigmoid` 对 $\mathbf{A}_{ij}$ 进行概率映射，确保惩罚项的物理意义（连接强度 $\times$ 冲突程度）。

### 6.2 结构正则化 (Structural Regularization)

1.  **稀疏诱导 (Sparsity Induction)**: L1 范数，推动连接矩阵稀疏化。

$$
\mathcal{L}_{metabolic} = \|\mathbf{A}\|_1
$$

2.  **二值化约束 (Binarization Constraint)**: 采用 **基尼不纯度 (Gini Impurity)** 形式的代理损失，替代不稳定的对数熵，迫使连接概率趋向于 0 或 1。

$$
\mathcal{L}_{det} = \text{Mean}(\sigma(\mathbf{A}) \cdot (1 - \sigma(\mathbf{A})))
$$

* **代码对应**: `regularization.py`。

---

## 7. 优化与超参细节 (Optimization Implementation)

* **双速率优化 (Two-Speed Optimization)**:
    * **物理参数 ($\theta, \mathbf{U}, \mathbf{V}$)**: 学习率 `0.005`，使用 Weight Decay `0.01`。
    * **拓扑参数 ($\mathbf{A}$)**: 学习率 `0.05` (10倍速)，**无 Weight Decay**。
    * **控制论逻辑**: 结构参数（拓扑）的演变速率需快于功能参数（权重）的积累速率，以确保在过拟合发生前完成子空间的划分与隔离。

* **秩配额 (Rank Quota)**:
    * **公式**: 采用 **平方根缩放** 法则。
      $$
      r = \lfloor \frac{D_{base}}{\sqrt{N}} \rfloor
      $$
    * **实验设置**: 当 $N=16, D_{base}=1024$ 时，$r = 256$。这保证了子空间足够紧凑以产生竞争，又足够宽裕以容纳特征。

---

## 总结

CDSP-MoE 架构构建了一个基于**反馈控制**的自组织系统。

* **Space (空间)**: 共享的物理基座提供了参数复用的可能性。
* **Time (时间)**: 滞后的梯度信息提供了参数更新方向的冲突检测信号。
* **Structure (结构)**: 动态拓扑矩阵根据冲突信号与任务梯度进行自我重构（剪枝与强化），最终实现特定任务在特定子空间上的解耦与特化。
