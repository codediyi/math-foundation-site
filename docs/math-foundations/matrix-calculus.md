# 矩阵微分

> 矩阵微分的目标不是背复杂公式，而是看懂梯度如何穿过线性层、attention、softmax、残差和 LayerNorm。结构改进和训练稳定性分析都离不开它。

## 学习目标

- 从标量导数过渡到向量梯度、Jacobian 和矩阵梯度。
- 能手推 \(Y=XW\)、\(S=QK^\top\)、\(O=PV\) 的反向传播。
- 能用上游梯度理解链式法则，而不是只写最终答案。
- 能用 PyTorch autograd 验证手推结果。
- 能解释 softmax、残差、LayerNorm 的梯度直觉。

## 从最简单的导数开始

若：

\[
y=x^2,
\]

则：

\[
\frac{dy}{dx}=2x.
\]

若 loss 是 \(L=y\)，则 \(dL/dx=2x\)。但神经网络里 \(y\) 往往只是中间变量，真正的 loss 在后面，因此需要上游梯度。

## 上游梯度

假设：

\[
y=x^2,
\quad
L=3y.
\]

链式法则：

\[
\frac{dL}{dx}=\frac{dL}{dy}\frac{dy}{dx}=3\cdot 2x.
\]

\(dL/dy\) 就是后面传回来的上游梯度。矩阵微分里最重要的习惯是：先写上游梯度，再推当前变量梯度。

## 向量梯度

如果 \(x\in\mathbb{R}^D\)，loss 是标量 \(L\)：

\[
\nabla_xL=\left[\frac{\partial L}{\partial x_1},\ldots,\frac{\partial L}{\partial x_D}\right].
\]

梯度和 \(x\) shape 相同。

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
L = (x ** 2).sum()
L.backward()
print(x.grad)
```

## Jacobian

如果输出也是向量：

\[
y=f(x),\quad x\in\mathbb{R}^n,\ y\in\mathbb{R}^m,
\]

Jacobian：

\[
J_{ij}=\frac{\partial y_i}{\partial x_j}.
\]

它描述每个输出分量对每个输入分量的敏感度。Transformer 中完整 Jacobian 很大，通常不显式构造，但理解它有助于分析残差和归一化。

## 矩阵乘法反传：\(Y=XW\)

设：

\[
X\in\mathbb{R}^{B\times D},\quad W\in\mathbb{R}^{D\times M},\quad Y=XW.
\]

上游梯度：

\[
G_Y=\frac{\partial L}{\partial Y}\in\mathbb{R}^{B\times M}.
\]

梯度：

\[
G_X=G_YW^\top,
\quad
G_W=X^\top G_Y.
\]

shape 检查：

| 梯度 | 计算 | shape |
|---|---|---|
| \(G_X\) | `(B,M) @ (M,D)` | `(B,D)` |
| \(G_W\) | `(D,B) @ (B,M)` | `(D,M)` |

## autograd 验证

```python
import torch

torch.manual_seed(0)
X = torch.randn(3, 4, requires_grad=True)
W = torch.randn(4, 5, requires_grad=True)
Y = X @ W
L = (Y ** 2).sum()
L.backward()

with torch.no_grad():
    GY = 2 * Y
    print(torch.allclose(X.grad, GY @ W.T))
    print(torch.allclose(W.grad, X.T @ GY))
```

## attention logits 反传：\(S=QK^\top\)

单 head：

\[
Q,K\in\mathbb{R}^{T\times d},
\quad
S=QK^\top.
\]

上游梯度：

\[
G_S=\frac{\partial L}{\partial S}\in\mathbb{R}^{T\times T}.
\]

梯度：

\[
G_Q=G_SK,
\quad
G_K=G_S^\top Q.
\]

若：

\[
S=\frac{QK^\top}{\sqrt{d}},
\]

则梯度也乘 \(1/\sqrt{d}\)。这说明缩放不仅影响前向 logits，也影响反向梯度尺度。

## weighted sum 反传：\(O=PV\)

\[
P\in\mathbb{R}^{T\times T},\quad V\in\mathbb{R}^{T\times d},\quad O=PV.
\]

上游梯度 \(G_O\in\mathbb{R}^{T\times d}\)：

\[
G_P=G_OV^\top,
\quad
G_V=P^\top G_O.
\]

value 的梯度会被 attention 权重汇总，attention 权重也会从输出误差中得到信号。

## softmax 梯度直觉

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

Jacobian：

\[
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j).
\]

直觉：一个 logit 变大会提高自己的概率，同时压低其它位置概率。若 softmax 已经接近 one-hot，很多位置梯度会很小。

## 残差的梯度

残差：

\[
y=x+f(x).
\]

Jacobian：

\[
\frac{\partial y}{\partial x}=I+J_f.
\]

即使 \(f\) 的梯度路径不理想，残差中的 \(I\) 也提供直接路径。这是深层 Transformer 能训练的重要原因之一。

## LayerNorm 梯度直觉

LayerNorm：

\[
\operatorname{LN}(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
\]

难点在于 \(\mu\) 和 \(\sigma^2\) 都由 \(x\) 计算，所以每个维度的梯度不是独立的。

初学阶段重点记住：

- 均值项让梯度在特征维上相互耦合。
- 方差项改变梯度尺度。
- \(\epsilon\) 是数值稳定项。
- Pre-LN/Post-LN 会改变梯度路径。

## Transformer 对应关系

| 微分概念 | Transformer 中的对应 |
|---|---|
| 上游梯度 | loss 从输出层传回当前模块的信号 |
| 矩阵乘法反传 | 线性层、Q/K/V、FFN |
| softmax Jacobian | attention 权重反传 |
| 残差 Jacobian | 深层梯度直接路径 |
| LayerNorm 梯度 | 归一化和训练稳定性 |
| 缩放因子 | 前向 logits 与反向梯度尺度 |

## 常见误区

- 不写上游梯度，直接背最终梯度。
- 只看公式，不检查 shape。
- 混淆 \(G_YW^\top\) 和 \(W^\top G_Y\)。
- 忘记 attention 缩放会影响反传尺度。
- 认为 autograd 会算就不需要理解；调试 NaN 和结构改进时仍需要手推直觉。

## 检查问题

1. 为什么矩阵微分里必须写上游梯度？
2. \(Y=XW\) 中 \(G_X\) 和 \(G_W\) 的 shape 分别是什么？
3. \(S=QK^\top\) 中 \(G_Q\) 为什么等于 \(G_SK\)？
4. softmax 很尖时，梯度会有什么问题？
5. 残差中的 \(I\) 对梯度传播有什么帮助？
6. LayerNorm 为什么让特征维梯度耦合？

## 分层练习

| 层级 | 任务 |
|---|---|
| 基础 | 用 autograd 验证 \((x^2).sum()\) 的梯度 |
| 推导 | 手推 \(Y=XW+b\) 对 \(b\) 的梯度 |
| 实现 | 验证 \(S=QK^\top/\sqrt{d}\) 对 \(Q,K\) 的梯度 |
| 诊断 | 改变 logits 缩放倍数，观察梯度范数 |
| 迁移 | 整理 attention 前向和反向 shape flow |

## 阶段产出

整理一张“Transformer 常见反传公式卡”，包括 \(XW\)、\(QK^\top\)、softmax、\(PV\)、残差、LayerNorm。每个公式必须写 shape 和一句梯度直觉。

## 教材补充：把反传写成 shape 守恒

一个简单但有效的检查原则：某个变量的梯度 shape 必须和该变量 shape 一致。

| 变量 | shape | 梯度 shape |
|---|---|---|
| \(X\) | `(B,D)` | `(B,D)` |
| \(W\) | `(D,M)` | `(D,M)` |
| \(Q\) | `(B,H,T,d)` | `(B,H,T,d)` |
| \(S\) | `(B,H,T,T)` | `(B,H,T,T)` |

如果你推出来的梯度 shape 不一致，公式一定有问题。

## 手推任务：bias 梯度

线性层：

\[
Y=XW+b.
\]

其中 \(b\in\mathbb{R}^{M}\)，上游梯度 \(G_Y\in\mathbb{R}^{B\times M}\)。因为 bias 被 broadcast 到 batch 维：

\[
G_b=\sum_{i=1}^{B}G_{Y,i}.
\]

代码中通常是：

```python
Gb = GY.sum(dim=0)
```

## Notebook 实验：验证 attention 三段反传

```python
import torch

torch.manual_seed(0)
T, d = 4, 5
Q = torch.randn(T, d, requires_grad=True)
K = torch.randn(T, d, requires_grad=True)
V = torch.randn(T, d, requires_grad=True)
S = Q @ K.T / (d ** 0.5)
P = torch.softmax(S, dim=-1)
O = P @ V
L = (O ** 2).sum()
L.backward()
print(Q.grad.shape, K.grad.shape, V.grad.shape)
```

下一步可以把 softmax 暂时替换成 identity，手推 \(S\) 和 \(V\) 的梯度，再逐步加回 softmax。

## 易错案例：广播导致的梯度求和

如果前向中某个张量被 broadcast，反向时对应维度通常要 sum 回去。例如 bias 从 `(M,)` broadcast 到 `(B,M)`，所以梯度要对 batch 维求和。

同理，mask 通常不需要梯度；position bias 若是可学习参数，则要考虑哪些维度被共享，反向时就会在哪些维度聚合。

## 论文阅读提示

读到下面内容时，矩阵微分是底层语言：

| 论文内容 | 关注点 |
|---|---|
| gradient flow | Jacobian 连乘是否稳定 |
| residual scaling | \(I+J_f\) 的尺度如何变化 |
| normalization analysis | 均值/方差如何影响梯度 |
| attention backward | Q/K/V 梯度路径 |
| stop gradient | 哪条路径被切断 |
| straight-through | 前向离散，反向近似 |

## 章节小测

1. 为什么 bias 梯度要对 batch 维求和？
2. 如果 \(S=QK^\top/\sqrt{d}\)，缩放对 \(G_Q\) 有什么影响？
3. softmax 的 Jacobian 为什么不是对角矩阵？
4. residual scaling 改变的是前向值、反向梯度，还是二者都有？
5. stop-gradient 会如何改变计算图？

## 本章完成标准

- 能用 shape 守恒检查反传公式。
- 能手推线性层 bias 梯度。
- 能写出 attention 三段反传的变量路径。
- 能读懂论文中关于 gradient flow 的基本表述。

