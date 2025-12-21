# CDSP-MoE 技术架构白皮书 (Technical Architecture Document)

**项目名称**: Conflict-Driven Subspace Pruning Mixture-of-Experts (CDSP-MoE)

**核心理念**: 利用“滞后梯度”计算参数更新方向的冲突程度，将其作为结构演化的负反馈信号，替代传统 MoE 的负载均衡辅助损失，实现神经网络拓扑结构的自适应稀疏化与模块化。

---

## 1. 物理层：共享子空间基座 (Physical Subspace Backbone)

不同于传统 MoE 中专家权重矩阵的物理隔离，CDSP 采用“超完备共享参数空间”设计，所有专家共享同一组底层参数，通过掩码进行逻辑区分。

* **定义**: 物理基座由两个正交初始化的超大参数矩阵组成：

$$
W_{phys} = \{ W_{down} \in \mathbb{R}^{D_{model} \times D_{hidden}}, \quad W_{up} \in \mathbb{R}^{D_{hidden} \times D_{model}} \}
$$

其中 $D_{hidden} \gg D_{model}$ (代码中 $D_{hidden} = N_{experts} \times d_{exp}$)。

* **物理初始分区 ($\mathcal{P}$)**:
    * 系统维护一个固定索引缓冲区 `self.pi` ($\pi$)，用于定义专家的初始参数索引范围。
    * **初始化策略**: 块状对角 (Block Diagonal) 划分。专家 $i$ 初始对应的参数区间 $\mathcal{P}_i$ 为：

$$
\mathcal{P}_i = \left[ i \cdot \frac{D_{hidden}}{N}, \quad (i+1) \cdot \frac{D_{hidden}}{N} \right)
$$

---

## 2. 拓扑层：动态连接矩阵 (Topology Layer)

该层定义了逻辑专家与物理参数之间的连接权重，是一个可学习的加权有向图。

* **拓扑矩阵 ($\alpha$)**:
    * 参数 `self.alpha` ($A \in \mathbb{R}^{N \times N}$)。
    * **数学定义**: $\alpha_{ij}$ 表示逻辑专家 $i$ 对 初始归属于 $j$ 的物理参数区间 $\mathcal{P}_j$ 的**连接强度**。

* **结构初始化 (Structural Initialization)**:
    * 采用“全连接初始化，稀疏化演进”的策略。
    * **对角线 (Self-Loop)**: $\alpha_{ii} \approx 1.0$ (保证专家自身的参数稳定性)。
    * **非对角线 (Cross-Link)**: $\alpha_{ij} \sim \mathcal{N}(0, 0.01)$ (引入微弱的全局连接与噪声，允许训练初期的梯度全局流动)。
    * **代码对应**: `topology.py` 中的 `_reset_parameters`。

* **参数控制力投影 (Control Projection)**:
    专家 $i$ 对物理维度 $k$ 的实际激活强度 $I_{i,k}$ 计算如下：

$$
I_{i,k} = \sum_{j=0}^{N-1} \text{Sigmoid}(\alpha_{ij}) \cdot \mathbb{1}(k \in \mathcal{P}_j)
$$

(即：维度 $k$ 的激活程度取决于其所属的初始分区 $j$ 与当前专家 $i$ 之间的拓扑连接强度)。

---

## 3. 感知路由层 (Perceptive Routing Layer)

路由层负责根据输入特征分配计算资源。为防止模型仅依赖任务 ID 进行简单查表（Shortcut Learning），引入了对抗性掩码机制。

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

* **特征融合 (Feature Fusion)**:
    将归一化的图像特征与处理后的任务向量进行拼接：

$$
h_{in} = [ x_{norm} \oplus v_{task} ]
$$

($\oplus$ 表示向量拼接)。

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
    根据控制力 $I_{i, \cdot}$ 选取 Top-$R$ 个物理维度索引 $\Omega_i$ (秩配额 $R$):

$$
\Omega_i = \text{TopK}(I_{i, \cdot}, k=R)
$$

* **代码对应**: `topology.py` -> `get_subspace_indices`。

2. **稀疏计算 (Sparse Computation)**:
    仅提取 $\Omega_i$ 对应的参数切片进行矩阵乘法。

$$
y_i = W_{up}[:, \Omega_i] \cdot \sigma(W_{down}[\Omega_i, :] \cdot x)
$$

* **代码对应**: `backbone.py` -> `forward`。

3. **强度调制 (Strength Modulation)**:
    输出结果受拓扑连接强度的二次加权，确保弱连接对应的参数对输出贡献极小。
    * **代码对应**: `layer.py` 中 `token_outputs * gate_score`。

---

## 5. 梯度冲突优化 (Gradient Conflict Optimization)

这是 CDSP 的核心结构优化机制。利用滞后梯度检测参数更新方向的冲突，作为结构剪枝的依据。

### 5.1 梯度采样 (Gradient Sampling)

利用 PyTorch Hook 机制，在反向传播阶段捕获每个专家在物理基座上的梯度投影。

* **数据结构**: `expert_grads[i] = (GradTensor, PhysicalIndices)`。
* **时序逻辑**: `optimizer.zero_grad()` 位于 `forward` 之后、`backward` 之前。因此，Forward 阶段读取的 `expert_grads` 为 **Step t-1** 的梯度信息 $g^{(t-1)}$。

### 5.2 空间冲突度量 (Spatial Conflict Metric)

对于当前 Batch 中同时激活的一对专家 $(i, j)$，计算其在**物理参数重叠区域**的梯度余弦相似度。

1.  **参数交集**: $\mathcal{S}_{ij} = \Omega_i \cap \Omega_j$。若交集为空，冲突为 0。
2.  **梯度对齐**: 提取交集 $\mathcal{S}_{ij}$ 对应的梯度切片 $g_i[\mathcal{S}], g_j[\mathcal{S}]$。
3.  **余弦相似度计算**:

$$
\text{CosSim}_{ij} = \frac{g_i[\mathcal{S}] \cdot g_j[\mathcal{S}]}{\|g_i[\mathcal{S}]\| \|g_j[\mathcal{S}]\| + \epsilon}
$$

4.  **干扰惩罚项**:

$$
\mathcal{L}_{\text{conflict}} = \sum_{i,j} G_i(x) G_j(x) \cdot \text{ReLU}( -\text{CosSim}_{ij} )
$$

* **物理意义**: 当且仅当梯度方向**相反**（负余弦，即参数更新方向冲突）时，产生正向惩罚。正交或同向被视为参数复用或协同，不产生惩罚。
* **代码对应**: `conflict.py` -> `compute_sparse_aligned_cosine`。

---

## 6. 全局结构演化目标 (Structural Evolution Objective)

总损失函数通过多目标优化，强制系统向“低熵、低干扰、模块化”的结构收敛。

### 6.1 冲突驱动剪枝 (Conflict-Driven Pruning)

如果专家 $i$ 与专家 $j$ 存在高强度的连接 ($\alpha_{ij}$ 大)，且二者在参数更新上存在严重冲突 ($\mathcal{L}_{\text{conflict}}$ 大)，则通过梯度下降抑制连接强度 $\alpha_{ij}$。

$$
\mathcal{L}_{prune} = \sum_{i,j} \text{Sigmoid}(\alpha_{ij}) \cdot \mathcal{L}_{\text{conflict}}(i, j)
$$

* **权重**: $\lambda = 10.0$ (强约束)。

### 6.2 结构正则化 (Structural Regularization)

1.  **稀疏诱导 (Sparsity Induction)**: L1 范数，推动连接矩阵稀疏化。

$$
\mathcal{L}_{L1} = \sum_{i,j} |\alpha_{ij}|
$$

2.  **二值化约束 (Binarization Constraint)**: 最小化二元熵，迫使连接权重趋向于 0 或 1 的确定性状态。

$$
\mathcal{L}_{entropy} = - \sum_{i,j} [p_{ij}\log p_{ij} + (1-p_{ij})\log(1-p_{ij})]
$$

其中 $p_{ij} = \text{Sigmoid}(\alpha_{ij})$。

* **代码对应**: `regularization.py`。

---

## 7. 优化与超参细节 (Optimization Implementation)

* **双速率优化 (Two-Speed Optimization)**:
    * **物理参数 ($W_{phys}$)**: 学习率 `0.005`，使用 Weight Decay `0.01`。
    * **拓扑参数 ($\alpha$)**: 学习率 `0.05` (10倍速)，**无 Weight Decay**。
    * **控制论逻辑**: 结构参数（拓扑）的演变速率需快于功能参数（权重）的积累速率，以确保在过拟合发生前完成子空间的划分与隔离。

* **秩配额 (Rank Quota)**:
    * 公式: $R = \gamma \cdot \frac{D_{hidden}}{N}$。
    * 实验设置: $\gamma=1.5$，即 $R = 1.5 \times D_{base}$。
    * **作用**: 限制每个专家可访问的最大参数量，强制其在有限资源下进行特征选择。

---

## 总结

CDSP-MoE 架构构建了一个基于**反馈控制**的自组织系统。

* **Space (空间)**: 共享的物理基座提供了参数复用的可能性。
* **Time (时间)**: 滞后的梯度信息提供了参数更新方向的冲突检测信号。
* **Structure (结构)**: 动态拓扑矩阵根据冲突信号进行自我重构（剪枝与强化），最终实现特定任务在特定子空间上的解耦与特化。