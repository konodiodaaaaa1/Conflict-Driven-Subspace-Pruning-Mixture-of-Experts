# CDSP-MoE 技术架构白皮书

**项目名称**: Conflict-Driven Subspace Pruning Mixture-of-Experts (CDSP-MoE)

**核心理念**: 基于“滞后梯度”检测参数更新方向的冲突，将其作为系统演化的负反馈信号。通过结构化的稀疏性诱导，替代传统 MoE 的外在负载均衡损失，实现神经网络拓扑结构的自适应特化与模块化。

---

## 1. 物理层：共享子空间基座 (Physical Subspace Backbone)

CDSP 采用“超完备共享参数空间”设计，所有专家共享同一组底层参数基座，通过逻辑掩码定义其可访问的操作区域。

* **定义**: 物理基座由两个正交初始化的参数矩阵组成：

$$
\mathbf{U}_{\text{base}} \in \mathbb{R}^{D_{\text{model}} \times D_{\text{base}}}, \quad \mathbf{V}_{\text{base}} \in \mathbb{R}^{D_{\text{base}} \times D_{\text{model}}}
$$

其中 $D_{\text{base}} \gg D_{\text{model}}$。代码默认配置中 $D_{\text{base}} = 4 \times D_{\text{model}}$。

* **物理索引分区 ($\mathbf{\Pi}$)**:
    系统维护一个固定的二进制掩码矩阵 `self.pi` ($\mathbf{\Pi} \in \{0,1\}^{N \times D_{\text{base}}}$)，用于定义专家的初始参数索引范围。
    * **分区策略**: 块状对角 (Block Diagonal) 划分。专家 $i$ 初始对应的参数区间为 $[i \cdot B, (i+1) \cdot B)$，其中块大小 $B = \lfloor D_{\text{base}}/N \rfloor$。

---

## 2. 拓扑层：动态连接矩阵 (Topology Layer)

该层定义了逻辑专家与物理参数之间的连接权重，本质上是一个可学习的加权有向图。

* **拓扑矩阵 ($\mathbf{A}$)**:
    对应代码参数 `self.alpha` ($\mathbf{A} \in \mathbb{R}^{N \times N}$)。
    * **数学定义**: $\mathbf{A}_{ij}$ 表示逻辑专家 $i$ 对初始归属于专家 $j$ 的物理参数分区的**连接强度**。

* **结构初始化 (Structural Initialization)**:
    采用“全连接初始化，稀疏化演进”策略：
    1.  **全局弱连接**: $\mathbf{A}_{ij} \leftarrow 0.1$。初始化为全通滤波器，允许梯度在初期全局流动。
    2.  **噪声注入**: $\mathbf{A}_{ij} \leftarrow \mathbf{A}_{ij} + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, 0.02)$，引入非对称性以打破梯度同步。
    3.  **对角线强化**: $\mathbf{A}_{ii} \leftarrow 1.0$。保证专家对自身初始分配的参数块具有绝对控制权。

* **控制力投影 (Control Projection)**:
    专家 $i$ 对物理维度 $k$ 的实际激活强度向量 $\mathbf{I}_{i}$ 计算如下：

$$
\mathbf{I}_{i} = \sigma(\mathbf{A}_{i, :}) \cdot \mathbf{\Pi}
$$

其中 $\sigma$ 为 Sigmoid 函数。

---

## 3. 感知路由层 (Perceptive Routing Layer)

路由层负责根据输入特征分配计算资源。为防止模型仅依赖任务 ID 进行简单查表（Shortcut Learning），引入了随机掩码机制。

* **任务特征掩码 (Task Masking)**:
    引入随机掩码变量 $\mathcal{M}$。在训练过程中，以概率 $p_{\text{drop}}$ 将任务嵌入向量置零，强制 Router 挖掘输入内容 $x_{\text{norm}}$ 的内在特征。

$$
v_{\text{task}} = \mathcal{M} \cdot \text{Embed}(t)
$$

其中掩码 $\mathcal{M}$ 的生成逻辑为：

$$
\mathcal{M} = \begin{cases} 
0 & \text{if } t \text{ is None} \lor (\text{Training} \land \xi < 0.1) \\ 
1 & \text{otherwise} 
\end{cases}
$$

* **特征融合 (Feature Fusion)**:
    将归一化的内容特征与处理后的任务向量进行拼接：

$$
h_{\text{in}} = [ \text{LayerNorm}(x) \oplus v_{\text{task}} ]
$$

* **门控决策 (Gating Decision)**:
    Router 基于融合特征计算专家的激活权重：

$$
G(x, t) = \text{Softmax}(W_{g} \cdot h_{\text{in}})
$$

---

## 4. 前向传播动力学 (Forward Dynamics)

当专家 $i$ 被激活时，通过动态索引机制调用物理基座的特定子集进行计算。

1.  **子空间寻址 (Addressing)**:
    根据控制力 $\mathbf{I}_{i}$ 选取 Top-$r$ 个物理维度索引 $\mathcal{S}_{i}$。
    * **秩配额 (Rank Quota)**: 代码定义为 $r = \lfloor D_{\text{base}} / \sqrt{N} \rfloor$。

$$
\mathcal{S}_{i}, \text{Strength}_{i} = \text{TopK}(\mathbf{I}_{i}, r)
$$

2.  **稀疏计算 (Sparse Computation)**:
    仅提取 $\mathcal{S}_{i}$ 对应的参数切片进行矩阵乘法。

$$
y_{\text{raw}} = \text{SiLU}(x \cdot \mathbf{U}_{\text{base}}[\mathcal{S}_{i}]) \cdot \mathbf{V}_{\text{base}}[\mathcal{S}_{i}]^{T}
$$

3.  **强度调制 (Strength Modulation)**:
    输出结果受拓扑连接强度的均值加权。若连接权重 $\mathbf{A}_{ij}$ 较低，对应的物理参数输出将被抑制。

$$
\text{GateScore} = \text{Mean}(\text{Strength}_{i})
$$

$$
y_{\text{final}} = y_{\text{raw}} \cdot \text{GateScore} \cdot \text{RouterWeight}
$$

---

## 5. 梯度冲突优化 (Gradient Conflict Optimization)

这是 CDSP 的核心结构优化机制。利用滞后梯度检测参数更新方向的冲突，作为结构剪枝的依据。

### 5.1 梯度采样 (Gradient Sampling)
利用 PyTorch Hook 机制，在反向传播阶段捕获每个专家在物理基座上的梯度投影。
* **数据结构**: `expert_grads[i] = (GradTensor, PhysicalIndices)`。
* **时序逻辑**: `optimizer.zero_grad()` 位于 `forward` 之后、`backward` 之前。因此，Forward 阶段读取的 `expert_grads` 为 **Step t-1** 的梯度信息。

### 5.2 空间冲突度量 (Spatial Conflict Metric)
对于当前 Batch 中同时激活的一对专家 $(i, j)$，计算其在**物理参数重叠区域**的梯度余弦相似度。

1.  **参数交集**: $\mathcal{K} = \mathcal{S}_{i} \cap \mathcal{S}_{j}$。若交集为空，冲突为 0。
2.  **梯度对齐**: 提取交集 $\mathcal{K}$ 对应的梯度切片 $g_{i}^{\mathcal{K}}$ 和 $g_{j}^{\mathcal{K}}$。
3.  **余弦相似度计算**:

$$
\cos(i, j) = \frac{g_{i}^{\mathcal{K}} \cdot g_{j}^{\mathcal{K}}}{\|g_{i}^{\mathcal{K}}\| \|g_{j}^{\mathcal{K}}\| + \epsilon}
$$

4.  **干扰惩罚项**:

$$
\text{Conflict}_{ij} = \text{ReLU}(-\cos(i, j))
$$

* **物理意义**: 当且仅当梯度方向**相反**（负余弦，即参数更新方向冲突）时，产生正向惩罚。

---

## 6. 全局结构演化目标 (Structural Evolution Objective)

总损失函数通过多目标优化，强制系统向“低熵、低干扰、模块化”的结构收敛。

### 6.1 冲突驱动剪枝 (Conflict-Driven Pruning)
如果专家 $i$ 与专家 $j$ 存在高强度的连接，且二者在参数更新上存在严重冲突，则通过梯度下降抑制连接强度。

$$
\mathcal{L}_{\text{conflict}} = \sum_{i \neq j} \sigma(\mathbf{A}_{ij}) \cdot \text{Conflict}_{ij}
$$

* **权重**: $\lambda_{\text{conf}} = 10.0$ (强约束)。
* **注意**: 公式中的 $\sigma$ (Sigmoid) 确保了只有活跃的连接才会受到冲突惩罚，避免对已断开的连接进行无效优化。

### 6.2 结构正则化 (Structural Regularization)

1.  **稀疏诱导 (Sparsity Induction)**: L1 范数，推动连接矩阵稀疏化。

$$
\mathcal{L}_{\text{L1}} = \text{Mean}(|\mathbf{A}|)
$$

2.  **二值化约束 (Binarization Constraint)**: 使用基尼不纯度 (Gini Impurity) 形式，迫使连接权重趋向于 0 或 1 的确定性状态。

$$
p = \sigma(\mathbf{A})
$$

$$
\mathcal{L}_{\text{det}} = \text{Mean}(p \cdot (1 - p))
$$

---

## 7. 优化与超参细节 (Optimization Implementation)

* **双速率优化 (Two-Speed Optimization)**:
    * **物理参数 ($\mathbf{U}, \mathbf{V}$)**: 学习率 `0.005`，使用 Weight Decay `0.01`。
    * **拓扑参数 ($\mathbf{A}$)**: 学习率 `0.05` (10倍速)，**无 Weight Decay**。
    * **控制逻辑**: 结构参数（拓扑）的演变速率需快于功能参数（权重）的积累速率，以确保在过拟合发生前完成子空间的划分与隔离。

* **秩配额 (Rank Quota)**:
    * 公式: $r = \lfloor D_{\text{base}} / \sqrt{N} \rfloor$。
    * **作用**: 限制每个专家可访问的最大参数量，强制其在有限资源下进行特征选择。
