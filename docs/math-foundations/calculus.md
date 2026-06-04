# 微积分复习

> 微积分帮助你理解“局部变化”。优化、反向传播、学习率、梯度裁剪、Taylor 展开和训练稳定性，本质上都在讨论函数在当前点附近如何变化。

## 学习目标

- 理解极限、连续、导数、偏导、方向导数和 Taylor 展开。
- 能用局部线性化解释梯度下降为什么可能下降，也可能发散。
- 能用数值微分验证简单梯度。
- 能把链式法则连接到神经网络反向传播。
- 能解释 Transformer 中学习率、warmup、残差、归一化与局部变化尺度的关系。

## 从一个最小例子开始

考虑损失函数：

\[
L(w)=(w-3)^2.
\]

它的最低点在 \(w=3\)。如果当前 \(w=0\)，导数为：

\[
L'(w)=2(w-3),\quad L'(0)=-6.
\]

负导数说明：增大 \(w\) 会让 loss 降低。梯度下降更新：

\[
w \leftarrow w-\eta L'(w).
\]

```python
w = 0.0
lr = 0.1
for step in range(6):
    loss = (w - 3) ** 2
    grad = 2 * (w - 3)
    w = w - lr * grad
    print(step, round(w, 3), round(loss, 3))
```

微积分的核心不是算出这个导数，而是理解：导数给出当前位置附近最直接的下降信号。

## 极限：靠近时会发生什么

极限描述当输入逐渐靠近某个点时，函数值趋向哪里。

\[
\lim_{x\to a}f(x)=L.
\]

普通语言：\(x\) 不一定等于 \(a\)，但越来越靠近 \(a\) 时，\(f(x)\) 越来越靠近 \(L\)。

导数的定义依赖极限：

\[
f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.
\]

这说明导数是“小范围变化率”的极限。

## 连续：输入小变，输出也小变

连续直觉：输入变化很小，输出不会突然跳变。大多数神经网络层是分段连续或连续的，例如线性层、GELU、softmax、LayerNorm。

但有些操作不是处处光滑：

| 操作 | 问题 |
|---|---|
| ReLU | 0 点不可导，但实际训练可用次梯度 |
| top-k | 选择集合会突然变化 |
| argmax | 不可导 |
| clipping | 边界处不可光滑 |

这就是为什么很多论文会用 softmax、Gumbel-softmax、straight-through estimator 等方法替代硬选择。

## 导数：局部线性近似

如果 \(h\) 很小：

\[
f(x+h)\approx f(x)+f'(x)h.
\]

这就是一阶 Taylor 展开。它说：函数在一个很小邻域内，可以用一条直线近似。

在多维参数 \(\theta\) 上：

\[
L(\theta+\Delta)\approx L(\theta)+\nabla L(\theta)^\top\Delta.
\]

若选择：

\[
\Delta=-\eta\nabla L(\theta),
\]

则局部线性项变化为：

\[
-\eta\|\nabla L(\theta)\|^2.
\]

所以在局部近似有效、学习率不太大时，loss 有下降趋势。

## 为什么学习率过大会发散

一阶近似只在局部有效。学习率太大时，更新一步可能跳出局部邻域，导致 loss 增大。

用二次函数看更清楚：

\[
L(w)=w^2,
\quad
w_{t+1}=w_t-2\eta w_t=(1-2\eta)w_t.
\]

若 \(|1-2\eta|<1\)，会收敛；若 \(\eta>1\)，会发散。

```python
for lr in [0.1, 0.9, 1.1]:
    w = 5.0
    hist = []
    for _ in range(6):
        hist.append(round(w, 3))
        w = w - lr * 2 * w
    print('lr=', lr, hist)
```

## 偏导：多参数函数的局部变化

深度学习中的损失函数依赖很多参数：

\[
L=L(w_1,w_2,\ldots,w_n).
\]

偏导：

\[
\frac{\partial L}{\partial w_i}
\]

表示只改变 \(w_i\)，其它参数暂时固定时，loss 的变化率。

梯度：

\[
\nabla L=\left[\frac{\partial L}{\partial w_1},\ldots,\frac{\partial L}{\partial w_n}\right].
\]

梯度是最陡上升方向，负梯度是局部最陡下降方向。

## 方向导数：沿某个方向变化

方向导数描述函数沿方向 \(v\) 的变化：

\[
D_v L(\theta)=\nabla L(\theta)^\top v.
\]

如果 \(v\) 与梯度方向一致，变化最大；如果与梯度垂直，局部一阶变化为 0。

这对理解优化很有用：模型参数空间里，有些方向非常敏感，有些方向变化很慢。条件数大时，优化会在陡峭方向震荡、在平坦方向前进缓慢。

## 二阶 Taylor 展开

一维：

\[
f(x+h)\approx f(x)+f'(x)h+\frac{1}{2}f''(x)h^2.
\]

多维：

\[
L(\theta+\Delta)\approx L(\theta)+\nabla L^\top\Delta+\frac{1}{2}\Delta^\top H\Delta.
\]

其中 \(H\) 是 Hessian。二阶项描述曲率。曲率大时，同样的学习率会带来更剧烈的 loss 变化。

## 数值微分验证梯度

数值微分：

\[
f'(x)\approx\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}.
\]

```python
def f(x):
    return x ** 3 + 2 * x

x = 1.5
eps = 1e-5
num_grad = (f(x + eps) - f(x - eps)) / (2 * eps)
true_grad = 3 * x ** 2 + 2
print(num_grad, true_grad)
```

在实现新 layer 或自定义 loss 时，数值梯度检查是非常实用的调试方法。

## 链式法则与反向传播

若：

\[
z=f(y),\quad y=g(x),
\]

则：

\[
\frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}.
\]

反向传播中，\(dz/dy\) 是上游梯度，\(dy/dx\) 是当前局部导数。每个模块只需要知道自己的局部导数，就能把梯度传回去。

## Transformer 对应关系

| 微积分概念 | Transformer 中的对应 |
|---|---|
| 导数 | 参数更新方向 |
| 偏导 | 每个权重矩阵元素的梯度 |
| 链式法则 | 多层 attention/FFN 的反向传播 |
| 一阶 Taylor | 梯度下降局部下降解释 |
| 二阶 Taylor | 曲率、学习率稳定区间 |
| 非光滑点 | ReLU、top-k、mask、clip |
| 数值微分 | 自定义算子梯度检查 |

## 逐步例题：softmax 平移不变性

softmax：

\[
\operatorname{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

对所有 logits 减同一个常数 \(c\)：

\[
\frac{e^{z_i-c}}{\sum_j e^{z_j-c}}
=\frac{e^{-c}e^{z_i}}{e^{-c}\sum_j e^{z_j}}
=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

所以减去最大值不改变 softmax 输出，但可以避免 `exp` overflow。

```python
import torch

z = torch.tensor([1000.0, 1001.0, 1002.0])
p = torch.softmax(z - z.max(), dim=0)
print(p)
```

## 常见误区

- 把会求导等同于懂微积分；真正重要的是局部变化直觉。
- 忘记 Taylor 近似有范围，学习率过大时不可靠。
- 认为所有操作都光滑可导；top-k、argmax、clip 都需要小心。
- 只看梯度大小，不看曲率和参数尺度。
- 用数值微分时 \(\epsilon\) 过大或过小，导致误差异常。

## 检查问题

1. 一阶 Taylor 展开为什么能解释梯度下降？
2. 学习率过大为什么可能让 loss 上升？
3. 偏导和梯度是什么关系？
4. 链式法则如何对应反向传播里的上游梯度？
5. softmax 为什么可以减去最大值？

## 分层练习

| 层级 | 任务 |
|---|---|
| 基础 | 用表格比较极限、连续、导数、偏导 |
| 推导 | 推导 \(L(w)=(w-3)^2\) 的梯度下降更新式 |
| 实现 | 用数值微分验证 \(x^3+2x\) 的导数 |
| 诊断 | 比较不同学习率在 \(w^2\) 上的收敛轨迹 |
| 迁移 | 解释 warmup 为什么能降低训练早期发散风险 |

## 阶段产出

写一页“微积分到训练稳定性”笔记，包含：一阶 Taylor、二阶曲率、学习率、链式法则、softmax 平移不变性。每个概念都配一个最小公式和一个代码观察。

## 教材补充：局部近似的边界

Taylor 展开经常被用来解释优化，但它不是全局真理。一阶近似：

\[
L(\theta+\Delta)\approx L(\theta)+\nabla L^\top\Delta
\]

只在 \(\Delta\) 足够小时可靠。二阶项很大时，同样的 \(\Delta\) 可能导致预测错误。

这就是为什么训练时要关注：

- 学习率是否太大。
- 梯度范数是否突然变大。
- 参数更新范数是否异常。
- loss landscape 是否存在尖锐方向。

## 手推任务：二次函数收敛条件

设：

\[
L(w)=aw^2,
\quad a>0.
\]

梯度：

\[
L'(w)=2aw.
\]

梯度下降：

\[
w_{t+1}=w_t-2a\eta w_t=(1-2a\eta)w_t.
\]

收敛要求：

\[
|1-2a\eta|<1.
\]

所以：

\[
0<\eta<\frac{1}{a}.
\]

曲率 \(a\) 越大，可用学习率越小。这是“尖锐方向难优化”的最小模型。

## Notebook 实验：数值梯度检查

```python
import torch

def f(x):
    return (x ** 4 + 2 * x ** 2).sum()

x = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
y = f(x)
y.backward()

eps = 1e-4
num = []
for i in range(x.numel()):
    xp = x.detach().clone(); xm = x.detach().clone()
    xp[i] += eps; xm[i] -= eps
    num.append((f(xp) - f(xm)) / (2 * eps))
print('autograd:', x.grad)
print('numeric:', torch.stack(num))
```

如果两者差距很大，先检查公式、dtype、eps、是否有不可导操作。

## Transformer 阅读提示

当论文讨论 stability、smoothness、Lipschitz、gradient flow 时，可以回到本章：

| 论文术语 | 本章解释 |
|---|---|
| local approximation | Taylor 展开 |
| gradient flow | 链式法则下的梯度传播 |
| sharp minima | 曲率较大的区域 |
| Lipschitz bound | 输入变化对输出变化的上界 |
| non-smooth operation | top-k、argmax、clip 等操作 |

## 章节小测

1. 为什么一阶 Taylor 能解释梯度下降方向？
2. 二阶项什么时候不能忽略？
3. 为什么曲率越大，学习率越要小？
4. 数值微分的 \(\epsilon\) 为什么不能随便取？
5. Transformer 中哪些操作不是标准光滑函数？

## 本章完成标准

- 能手推二次函数的梯度下降收敛条件。
- 能用数值微分检查一个向量函数梯度。
- 能解释学习率、梯度范数、曲率三者关系。
- 能把 Taylor 展开用于解释一次训练更新。

