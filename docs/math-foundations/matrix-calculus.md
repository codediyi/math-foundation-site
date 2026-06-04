# 矩阵微分

> 矩阵微分的目标不是背一堆复杂公式，而是看懂梯度如何穿过线性层、attention、softmax、残差和 LayerNorm。

## 学习目标

- 从标量导数过渡到向量梯度、Jacobian 和矩阵梯度。
- 能手推 \(Y=XW\)、\(S=QK^\top\)、\(O=PV\) 的反向传播。
- 能用上游梯度理解链式法则，而不是只写最终答案。
- 能用 PyTorch autograd 验证手推结果。

## 先从最简单的导数开始

如果：

\[
y=x^2
\]

那么：

\[
\frac{dy}{dx}=2x
\]

如果 loss 是 \(L=y\)，那 \(\frac{dL}{dx}=2x\)。但神经网络里 \(y\) 往往只是中间变量，真正的 loss 在后面。因此需要上游梯度。

## 什么是上游梯度

假设：

\[
y=x^2,\quad L=3y
\]

链式法则：

\[
\frac{dL}{dx}=\frac{dL}{dy}\frac{dy}{dx}=3\cdot 2x
\]

\(\frac{dL}{dy}\) 就是从后面传回来的上游梯度。矩阵微分里最重要的习惯是：先写上游梯度，再推当前变量梯度。

## 向量梯度

如果 \(x\in\mathbb{R}^D\)，loss 是标量 \(L\)，梯度：

\[
\nabla_x L=\left[\frac{\partial L}{\partial x_1},\ldots,\frac{\partial L}{\partial x_D}\right]
\]

梯度和 \(x\) 形状相同。代码中，如果 `x.shape == (D,)`，那么 `x.grad.shape` 也通常是 `(D,)`。

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
L = (x ** 2).sum()
L.backward()
print(x.grad)  # 2*x
```

## Jacobian

如果输出也是向量：

\[
y=f(x),\quad x\in\mathbb{R}^n,\ y\in\mathbb{R}^m
\]

Jacobian：

\[
J_{ij}=\frac{\partial y_i}{\partial x_j}
\]

它表示每个输出分量对每个输入分量的敏感度。Transformer 中完整 Jacobian 很大，通常不会显式构造，但理解它有助于分析残差和归一化。

## 矩阵乘法反传：\(Y=XW\)

设：

\[
X\in\mathbb{R}^{B\times D},\quad W\in\mathbb{R}^{D\times M},\quad Y=XW
\]

上游梯度：

\[
G_Y=\frac{\partial L}{\partial Y}\in\mathbb{R}^{B\times M}
\]

梯度公式：

\[
G_X=G_YW^\top
\]

\[
G_W=X^\top G_Y
\]

检查 shape：

| 梯度 | 计算 | shape |
|---|---|---|
| \(G_X\) | `(B,M) @ (M,D)` | `(B,D)` |
| \(G_W\) | `(D,B) @ (B,M)` | `(D,M)` |

这个公式是线性层、Q/K/V 投影、FFN 的基本反传单元。

## 用 autograd 验证

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
    GX = GY @ W.T
    GW = X.T @ GY
    print(torch.allclose(X.grad, GX))
    print(torch.allclose(W.grad, GW))
```

这段代码的意义不是展示 PyTorch 很强，而是确认手推公式和自动求导一致。

## attention logits 反传：\(S=QK^\top\)

单 head 下：

\[
Q,K\in\mathbb{R}^{T\times d},\quad S=QK^\top
\]

上游梯度：

\[
G_S=\frac{\partial L}{\partial S}\in\mathbb{R}^{T\times T}
\]

梯度：

\[
G_Q=G_SK
\]

\[
G_K=G_S^\top Q
\]

检查 shape：

| 梯度 | 计算 | shape |
|---|---|---|
| \(G_Q\) | `(T,T) @ (T,d)` | `(T,d)` |
| \(G_K\) | `(T,T).T @ (T,d)` | `(T,d)` |

如果 logits 有缩放：

\[
S=\frac{QK^\top}{\sqrt{d}}
\]

那么 \(G_Q\) 和 \(G_K\) 也要乘 \(1/\sqrt{d}\)。

## weighted sum 反传：\(O=PV\)

attention 输出：

\[
O=PV
\]

其中：

\[
P\in\mathbb{R}^{T\times T},\quad V\in\mathbb{R}^{T\times d},\quad O\in\mathbb{R}^{T\times d}
\]

上游梯度 \(G_O\in\mathbb{R}^{T\times d}\)：

\[
G_P=G_OV^\top
\]

\[
G_V=P^\top G_O
\]

这说明 value 的梯度会被 attention 权重 \(P\) 汇总，attention 权重本身也会从输出误差中得到信号。

## softmax 的梯度直觉

softmax：

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Jacobian：

\[
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)
\]

直觉：

- 一个 logit 变大，会提高自己的概率。
- 由于概率和为 1，它也会压低其它位置概率。
- 如果 softmax 已经非常接近 one-hot，很多位置梯度会很小。

这就是 logits 尺度和 attention 缩放会影响训练稳定性的原因之一。

## 残差的梯度

残差结构：

\[
y=x+f(x)
\]

Jacobian：

\[
\frac{\partial y}{\partial x}=I+J_f
\]

直觉：即使 \(f\) 的梯度路径不理想，残差中的 \(I\) 也提供了一条直接路径。这是深层 Transformer 能训练的重要原因之一。

## LayerNorm 为什么难

LayerNorm：

\[
\operatorname{LN}(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
\]

难点在于 \(\mu\) 和 \(\sigma^2\) 都由 \(x\) 计算得到，所以每个维度的梯度不是独立的。学习初期不需要背完整反传公式，但要知道：

- 均值项会让梯度在特征维上相互耦合。
- 方差项会改变梯度尺度。
- \(\epsilon\) 是数值稳定项，不是可有可无。

## Transformer 对应关系

| 微分概念 | Transformer 中的对应 |
|---|---|
| 上游梯度 | loss 从输出层传回当前模块的信号 |
| 矩阵乘法反传 | 线性层、Q/K/V、FFN |
| softmax Jacobian | attention 权重反传 |
| 残差 Jacobian | 深层梯度直接路径 |
| LayerNorm 梯度 | 归一化和训练稳定性 |

## 常见误区

- 不写上游梯度，直接背最终梯度。
- 只看公式，不检查 shape。
- 混淆 \(G_YW^\top\) 和 \(W^\top G_Y\)。
- 忘记 attention 缩放会影响反传尺度。
- 认为 autograd 会算就不需要理解；调试 NaN 和结构改进时仍然需要手推直觉。

## 检查问题

1. 为什么矩阵微分里必须写上游梯度？
2. \(Y=XW\) 中 \(G_X\) 和 \(G_W\) 的 shape 分别是什么？
3. \(S=QK^\top\) 中 \(G_Q\) 为什么等于 \(G_SK\)？
4. softmax 很尖时，梯度会有什么问题？
5. 残差中的 \(I\) 对梯度传播有什么帮助？

## 最小练习

1. 手推 \(Y=XW+b\) 对 \(b\) 的梯度。
2. 手推 \(S=QK^\top/\sqrt{d}\) 对 \(Q,K\) 的梯度。
3. 手推 \(O=PV\) 对 \(P,V\) 的梯度。
4. 用 autograd 验证上述三个公式。
5. 改变 logits 缩放倍数，观察 softmax 梯度范数变化。

## 阶段产出

整理一张“Transformer 常见反传公式卡”，包括 \(XW\)、\(QK^\top\)、softmax、\(PV\)、残差、LayerNorm。每个公式必须写 shape 和一句梯度直觉。
