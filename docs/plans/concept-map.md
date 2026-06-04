# 数学概念到 Transformer 映射表

## 核心映射

| 数学概念 | Transformer 对应对象 | 最低掌握程度 | 最小练习 |
|---|---|---|---|
| 向量空间 | token embedding、hidden state | 区分坐标表示和抽象向量 | 解释 embedding 矩阵行/列含义 |
| 线性映射 | Q/K/V/O 投影、FFN | 知道矩阵乘法是空间变换 | 写出 `Y = X @ W` 形状流 |
| 秩与零空间 | 信息压缩、头冗余、低秩 attention | 解释 rank 低意味着什么 | 比较随机矩阵和低秩矩阵 rank |
| SVD | 低秩近似、KV cache 压缩 | 知道截断 SVD 的误差直觉 | rank-k 重构实验 |
| 方差 | attention logits 缩放 | 推导 \(q^\top k\) 方差 | 模拟不同 \(d_k\) 下 logits 方差 |
| 熵/KL | 交叉熵、蒸馏、采样温度 | 知道 KL 非对称 | 实现 CE/KL 并比较 |
| Jacobian | 残差、反传、归一化梯度 | 写出链式法则矩阵形式 | 推导 \(Y=XW\) 梯度 |
| 核方法 | Performer、线性 attention | 理解 softmax kernel 近似 | 实现随机特征 toy example |
| 随机矩阵 | 初始化、谱稳定、宽度极限 | 能读奇异值分布 | 画不同宽度权重谱 |
| 群作用/旋转 | RoPE、相对位置相位 | 把 RoPE 写成分块旋转 | 验证相对位置依赖 |

## 公式中的依赖

### Scaled Dot-Product Attention

\[
S=\frac{QK^\top}{\sqrt{d_k}},\quad P=\operatorname{softmax}(S+M),\quad O=PV.
\]

需要：矩阵乘法、方差、softmax、mask、矩阵微分。

### Multi-Head Attention

\[
\operatorname{MHA}(X)=\operatorname{Concat}(O_1,\ldots,O_H)W^O.
\]

需要：子空间投影、张量 reshape、concat、线性映射。

### Training Stability

\[
x_{l+1}=x_l+f(\operatorname{LN}(x_l)).
\]

需要：残差 Jacobian、归一化、优化尺度、梯度传播。

## 进入论文阅读前的最低门槛

| 论文线 | 数学门槛 | 实现门槛 |
|---|---|---|
| RoPE / ALiBi | 旋转、内积、相对位置 | 实现位置 bias 或 RoPE |
| FlashAttention | softmax、log-sum-exp、分块矩阵乘法 | 实现普通 attention 和 online softmax |
| Pre-LN / DeepNorm | 残差、Jacobian、梯度尺度 | 比较 Pre-LN/Post-LN 曲线 |
| Performer / GLA | 核近似、递推状态 | 写一个线性 attention toy |
| Retrieval Heads | attention head、干预实验 | 做 head ablation 或 cache pruning |

