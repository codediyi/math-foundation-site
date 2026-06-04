# 张量运算

> 张量运算的核心不是“几维数组”，而是每一轴的语义。Transformer 里的大部分实现 bug 都不是数学公式错，而是 batch、head、sequence、feature 轴对错了。

## 学习目标

- 能解释标量、向量、矩阵、三维张量、四维张量的 shape 和轴语义。
- 能熟练读写 `reshape`、`view`、`transpose`、`permute`、`contiguous`、broadcast。
- 能用 `matmul` 和 `einsum` 写 attention logits。
- 能设计 padding mask、causal mask、position bias 的可广播 shape。
- 能画出 Transformer 从 embedding 到 attention 输出的 shape flow。

## 从一个最小例子开始

语言模型输入通常是 token 序列。embedding 后得到：

\[
X\in\mathbb{R}^{B\times T\times D}.
\]

读成：`B` 个样本，每个样本 `T` 个 token，每个 token 是 `D` 维向量。

```python
import torch

B, T, D = 2, 5, 16
X = torch.randn(B, T, D)
print(X.shape)
```

如果只说“这是三维张量”，信息是不够的。必须说明每一轴是什么。

## 常见轴语义

| 符号 | 代码名 | 含义 |
|---|---|---|
| \(B\) | `batch` | 一次并行多少样本 |
| \(T\) 或 \(N\) | `seq_len` | 序列长度/token 数 |
| \(D\) | `d_model` | hidden size |
| \(H\) | `num_heads` | attention head 数 |
| \(d_h\) | `head_dim` | 每个 head 的维度 |
| \(V\) | `vocab_size` | 词表大小 |

常见 shape：

| 张量 | shape | 含义 |
|---|---|---|
| token ids | `(B,T)` | 每个位置的 token 编号 |
| hidden states | `(B,T,D)` | 每个 token 的向量表示 |
| Q/K/V before split | `(B,T,D)` | 线性投影后表示 |
| Q/K/V after split | `(B,H,T,d_h)` | 多头表示 |
| attention logits | `(B,H,T,T)` | 每个 query 对每个 key 的分数 |
| attention weights | `(B,H,T,T)` | softmax 后权重 |
| attention output | `(B,H,T,d_h)` | 每个 head 的输出 |

## reshape：改变形状，不自动改变语义

多头注意力要把 \(D\) 拆成 \(H\times d_h\)：

```python
B, T, D, H = 2, 5, 16, 4
dh = D // H
X = torch.randn(B, T, D)
Q = X.reshape(B, T, H, dh).transpose(1, 2)
print(Q.shape)  # (B, H, T, dh)
```

关键步骤：

1. `(B,T,D)` 变成 `(B,T,H,dh)`。
2. `transpose(1, 2)` 变成 `(B,H,T,dh)`。
3. 后续 attention 在最后两维做矩阵乘法。

## transpose 与 permute

`transpose(i, j)` 交换两个维度。`permute` 重新排列所有维度。

```python
X = torch.randn(2, 5, 4, 8)  # (B,T,H,dh)
Y = X.transpose(1, 2)         # (B,H,T,dh)
Z = X.permute(0, 2, 1, 3)     # 同样是 (B,H,T,dh)
```

常见错误是把 `(B,T,H,dh)` 当成 `(B,H,T,dh)` 用，代码可能能跑，但 attention 语义完全错。

## contiguous 和 view

`transpose` 后的张量内存布局可能不连续。若要用 `view`，常需要：

```python
Y = X.transpose(1, 2).contiguous().view(2, 4, 5, 8)
```

更稳妥的初学写法是优先用 `reshape`，但仍要知道它可能触发拷贝。

## 矩阵乘法在高维张量中的规则

对于：

```python
Q.shape == (B, H, Tq, d)
K.shape == (B, H, Tk, d)
```

attention logits：

```python
S = Q @ K.transpose(-1, -2)
```

输出：

```python
S.shape == (B, H, Tq, Tk)
```

前面的 `B,H` 是批量维，矩阵乘法发生在最后两维：`(Tq,d) @ (d,Tk)`。

## einsum：把公式写成代码

指标形式：

\[
S_{b,h,i,j}=\sum_d Q_{b,h,i,d}K_{b,h,j,d}.
\]

代码：

```python
S = torch.einsum('bhid,bhjd->bhij', Q, K)
```

读法：输入都有 `d`，输出没有 `d`，说明对 `d` 求和；`i` 和 `j` 被保留，所以输出是 query-key 位置矩阵。

## broadcast：自动扩展维度

广播规则从右往左对齐维度。若某一维是 1，可以扩展到目标大小。

```python
X = torch.randn(2, 5, 16)
bias = torch.randn(16)
Y = X + bias
print(Y.shape)  # (2,5,16)
```

bias 的 shape `(16,)` 被广播到 `(2,5,16)`。

## mask 的 shape

attention logits shape 通常是 `(B,H,T,T)`。

### causal mask

```python
T = 5
causal = torch.tril(torch.ones(T, T)).bool()
logits = torch.randn(2, 4, T, T)
logits = logits.masked_fill(~causal, float('-inf'))
```

`(T,T)` 会广播到 `(B,H,T,T)`。

### padding mask

若 `valid` shape 是 `(B,T)`，表示哪些 token 有效，需要扩展为 `(B,1,1,T)`：

```python
B, H, T = 2, 4, 5
valid = torch.tensor([[1,1,1,0,0],[1,1,1,1,0]]).bool()
mask = valid[:, None, None, :]
logits = torch.randn(B, H, T, T)
logits = logits.masked_fill(~mask, float('-inf'))
```

这里 mask 作用在 key 位置，即最后一维。

## attention shape flow

| 步骤 | shape |
|---|---|
| 输入 hidden | `(B,T,D)` |
| 线性投影 Q/K/V | `(B,T,D)` |
| 拆 head | `(B,H,T,d_h)` |
| logits | `(B,H,T,T)` |
| mask 后 logits | `(B,H,T,T)` |
| softmax 权重 | `(B,H,T,T)` |
| weighted sum | `(B,H,T,d_h)` |
| 合并 head | `(B,T,D)` |
| 输出投影 | `(B,T,D)` |

## 最小 attention 代码

```python
import torch

B, T, D, H = 2, 5, 16, 4
dh = D // H
X = torch.randn(B, T, D)
Wq = torch.randn(D, D)
Wk = torch.randn(D, D)
Wv = torch.randn(D, D)

Q = (X @ Wq).reshape(B, T, H, dh).transpose(1, 2)
K = (X @ Wk).reshape(B, T, H, dh).transpose(1, 2)
V = (X @ Wv).reshape(B, T, H, dh).transpose(1, 2)

logits = Q @ K.transpose(-1, -2) / (dh ** 0.5)
weights = torch.softmax(logits, dim=-1)
out = weights @ V
out = out.transpose(1, 2).reshape(B, T, D)
print(out.shape)
```

## 与 Transformer 的关系

| 张量操作 | Transformer 中的位置 |
|---|---|
| reshape | 拆分/合并 multi-head |
| transpose | 调整矩阵乘法收缩轴 |
| broadcast | mask、bias、position bias |
| einsum | attention logits、特征交互 |
| softmax dim | query 对 key 的归一化 |
| contiguous | 高性能实现和 view 安全 |

## 常见误区

- 只看张量 rank，不写每一轴语义。
- 把 batch 轴或 head 轴错误地参与矩阵乘法。
- `softmax(dim=1)` 随手写，实际应该沿 key 维 `dim=-1`。
- padding mask 扩展到 query 维，而不是 key 维。
- `transpose` 后直接 `view`，忽略内存连续性。
- `einsum` 输出标签写错，代码能跑但含义错。

## 检查问题

1. `(B,T,D)` 拆成 `(B,H,T,d_h)` 需要哪两步？
2. `Q @ K.transpose(-1, -2)` 中矩阵乘法发生在哪两维？
3. padding mask 为什么通常扩成 `(B,1,1,T)`？
4. `einsum('bhid,bhjd->bhij')` 对哪个维度求和？
5. softmax 应该沿 logits 的哪一维做？为什么？

## 分层练习

| 层级 | 任务 |
|---|---|
| 基础 | 给 5 个 Transformer 张量写 shape 和轴语义 |
| 推导 | 把 \(S_{bhij}=\sum_dQ_{bhid}K_{bhjd}\) 翻译成 einsum |
| 实现 | 写最小 multi-head attention shape flow |
| 诊断 | 故意写错 mask shape，观察广播结果 |
| 迁移 | 读一个 attention 实现，标注每一行 shape |

## 阶段产出

画一张 Transformer shape flow 图，覆盖 token ids、embedding、Q/K/V、logits、mask、softmax、weighted sum、合并 head、输出投影。每个箭头标出使用的张量操作。

## 教材补充：从公式省略维度到代码完整维度

论文常写：

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d})V.
\]

代码里至少有四维：

\[
Q,K,V\in\mathbb{R}^{B\times H\times T\times d}.
\]

所以公式里的 \(QK^\top\) 实际是：

\[
S_{b,h,i,j}=\sum_dQ_{b,h,i,d}K_{b,h,j,d}.
\]

写代码前，必须把省略的 batch/head 维补回来。

## 手推任务：mask 广播检查

logits：

\[
S\in\mathbb{R}^{B\times H\times T_q\times T_k}.
\]

padding mask：

\[
M\in\mathbb{R}^{B\times T_k}.
\]

为了作用在 key 维，需要变成：

\[
M'\in\mathbb{R}^{B\times 1\times 1\times T_k}.
\]

这样会广播到每个 head、每个 query 位置。

## Notebook 实验：验证 matmul 与 einsum

```python
import torch

B, H, T, d = 2, 3, 4, 5
Q = torch.randn(B, H, T, d)
K = torch.randn(B, H, T, d)

s1 = Q @ K.transpose(-1, -2)
s2 = torch.einsum('bhid,bhjd->bhij', Q, K)
print(torch.allclose(s1, s2))
```

如果不相等，通常是维度顺序或转置写错。

## 易错案例：softmax 维度写错

```python
logits = torch.randn(2, 4, 5, 5)
wrong = torch.softmax(logits, dim=1)
right = torch.softmax(logits, dim=-1)
print(wrong.sum(dim=-1)[0, 0])
print(right.sum(dim=-1)[0, 0])
```

attention 权重应该对 key 维求和为 1，即最后一维。如果 `dim=1`，归一化发生在 head 维，语义错误。

## 论文阅读提示

高效 attention 论文中常见 shape 词汇：

| 术语 | shape 关注点 |
|---|---|
| block/chunk | 序列维被分块 |
| tile | GPU 计算块，常对应局部矩阵乘法 |
| KV cache | 通常缓存 `(B,H,T,d)` 的 K/V |
| grouped-query attention | Q head 数与 KV head 数不同 |
| sliding window | mask 只允许局部 key 可见 |
| prefix/past key values | 历史序列维不断增长 |

## 调试清单

- 每次 reshape 后立刻打印 shape。
- 每次 transpose 后确认轴语义。
- 每次 softmax 后检查归一化维度求和是否为 1。
- 每次 mask 后检查被 mask 的位置是否为 `-inf`。
- 合并 head 前确认 `(B,H,T,d)` 还是 `(B,T,H,d)`。
- 写 einsum 时确认消失的标签就是求和维。

## 章节小测

1. 为什么论文公式常省略 batch/head 维？
2. padding mask 和 causal mask 分别作用在哪些位置？
3. grouped-query attention 会改变哪些 head 维 shape？
4. KV cache 的序列维为什么会随生成增长？
5. FlashAttention 的分块主要针对哪两个维度？

## 本章完成标准

- 能把 attention 公式扩写成四维指标形式。
- 能写出 padding mask 和 causal mask 的可广播 shape。
- 能用 matmul/einsum 实现同一个 logits。
- 能独立画出 MHA 的完整 shape flow。

