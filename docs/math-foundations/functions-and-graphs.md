# 函数、图像与变化率

> 函数是深度学习里最基础的对象。模型是函数，损失是函数，训练是在调函数的参数；导数描述函数局部怎么变，梯度下降利用这个局部变化方向更新参数。

## 学习目标

- 能用普通语言解释函数、输入、输出、参数、图像和变化率。
- 能读懂线性函数、二次函数、指数、对数、sigmoid、softmax 的图像直觉。
- 能从斜率过渡到导数、偏导和梯度。
- 能解释链式法则为什么就是反向传播的数学核心。
- 能用 PyTorch/NumPy 画出函数值并观察梯度饱和。

## 从一个最小例子开始

假设你有一个非常简单的模型：

\[
\hat{y}=wx+b.
\]

这里有三个对象：输入 \(x\)、参数 \(w,b\)、输出 \(\hat{y}\)。如果 \(w=2,b=1\)，那么：

| 输入 \(x\) | 输出 \(2x+1\) |
|---:|---:|
| 0 | 1 |
| 1 | 3 |
| 2 | 5 |

这就是函数：给一个输入，按规则得到一个输出。深度学习模型只是把这个规则变得很大、很深、参数很多。

```python
import torch

x = torch.tensor([0.0, 1.0, 2.0])
w, b = 2.0, 1.0
y_hat = w * x + b
print(y_hat)
```

## 函数的基本读法

函数通常写成：

\[
y=f(x).
\]

读成：输入 \(x\)，经过规则 \(f\)，得到输出 \(y\)。

| 符号 | 含义 | 模型中的例子 |
|---|---|---|
| \(x\) | 输入 | token id、embedding、图像特征 |
| \(f\) | 映射规则 | Transformer block、MLP、attention |
| \(\theta\) | 参数 | 权重矩阵、bias、LayerNorm 参数 |
| \(f_\theta(x)\) | 带参数的函数 | 神经网络预测 |
| \(L(\theta)\) | 损失函数 | cross entropy、MSE |

学习函数时不要只看公式，还要问：输入是什么 shape，输出是什么 shape，参数在哪里。

## 图像怎么读

一维函数可以画成曲线。读图像时看四件事：

1. 输入范围是什么。
2. 输出范围是什么。
3. 函数是上升、下降还是有转折。
4. 哪些区域变化快，哪些区域变化慢。

例如 sigmoid：

\[
\sigma(x)=\frac{1}{1+e^{-x}}.
\]

它的输出永远在 0 到 1 之间。当 \(x\) 很大或很小时，曲线几乎变平，导数很小，这叫饱和。

## 常见函数直觉表

| 函数 | 公式 | 图像直觉 | 深度学习用途 |
|---|---|---|---|
| 线性函数 | \(ax+b\) | 一条直线 | 线性层、logits |
| 二次函数 | \(x^2\) | U 形曲线 | MSE、局部二阶近似 |
| 指数函数 | \(e^x\) | 增长越来越快 | softmax、概率归一化 |
| 对数函数 | \(\log x\) | 增长越来越慢 | log likelihood、cross entropy |
| sigmoid | \(1/(1+e^{-x})\) | S 形压缩 | 门控、二分类概率 |
| softmax | \(e^{z_i}/\sum_j e^{z_j}\) | 多个分数归一化 | attention 权重、词表分布 |

## 斜率：输出变化有多快

直线 \(y=ax+b\) 的斜率是 \(a\)。如果 \(a=2\)，输入增加 1，输出增加 2。

一般函数没有固定斜率，所以需要局部斜率：

\[
\frac{f(x+h)-f(x)}{h}.
\]

当 \(h\) 很小时，这个比值描述 \(x\) 附近的变化率。

## 导数：无限小范围内的斜率

导数定义：

\[
f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.
\]

普通语言：如果输入 \(x\) 只动一点点，输出会往哪个方向变、变多快。

常见导数：

| 函数 | 导数 | 直觉 |
|---|---|---|
| \(x^2\) | \(2x\) | 离 0 越远变化越快 |
| \(e^x\) | \(e^x\) | 指数函数变化率等于自身 |
| \(\log x\) | \(1/x\) | 小数附近变化更敏感 |
| \(ax+b\) | \(a\) | 线性函数斜率固定 |

## 偏导和梯度

模型参数不止一个。若损失函数为：

\[
L=L(w_1,w_2,\ldots,w_n),
\]

偏导 \(\partial L/\partial w_i\) 表示只改变 \(w_i\)，其它参数暂时固定时，损失如何变化。

梯度把所有偏导放在一起：

\[
\nabla L=\left[\frac{\partial L}{\partial w_1},\ldots,\frac{\partial L}{\partial w_n}\right].
\]

梯度的 shape 应该和参数 shape 一致。若 `W.shape == (D, M)`，通常 `W.grad.shape == (D, M)`。

## 链式法则：反向传播的核心

如果：

\[
y=f(g(x)),
\]

则：

\[
\frac{dy}{dx}=\frac{dy}{dg}\frac{dg}{dx}.
\]

普通语言：最终输出对输入的影响，等于中间每一段影响连乘。

神经网络是很多函数的复合：

\[
x \rightarrow h_1 \rightarrow h_2 \rightarrow \hat{y} \rightarrow L.
\]

反向传播就是从 \(L\) 开始，把局部导数一层层乘回来。

## softmax 为什么会饱和

softmax：

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

如果 logits 差距很大，例如 `[1, 2, 20]`，最大项的指数会远大于其它项，softmax 接近 one-hot。此时很多位置梯度会很小。

```python
import torch

for scale in [1.0, 5.0, 20.0]:
    z = torch.tensor([1.0, 2.0, 3.0]) * scale
    p = torch.softmax(z, dim=0)
    print(scale, p, 'max=', p.max().item())
```

## 数值观察：sigmoid 饱和

```python
import torch

x = torch.linspace(-8, 8, 9, requires_grad=True)
y = torch.sigmoid(x)
y.sum().backward()
print(torch.stack([x.detach(), y.detach(), x.grad], dim=1))
```

观察点：中间区域梯度较大，两端梯度接近 0。这解释了为什么早期深层网络容易出现梯度消失，也解释了为什么 Transformer 更常使用 ReLU/GELU 与残差路径。

## 与 Transformer 的关系

| 函数概念 | Transformer 中的位置 |
|---|---|
| 线性函数 | embedding 投影、Q/K/V/O、FFN |
| 非线性函数 | GELU、SiLU、门控激活 |
| 指数函数 | softmax attention 权重 |
| 对数函数 | cross entropy 和 log likelihood |
| 导数 | 反向传播和参数更新 |
| 链式法则 | 多层 Transformer 梯度传播 |
| 饱和 | softmax 过尖、梯度变小 |

## 逐步例题：从 MSE 到梯度下降

设：

\[
\hat{y}=wx,
\quad
L=(\hat{y}-y)^2.
\]

推导：

\[
\frac{\partial L}{\partial w}=2(\hat{y}-y)\frac{\partial \hat{y}}{\partial w}=2(wx-y)x.
\]

更新：

\[
w \leftarrow w-\eta\frac{\partial L}{\partial w}.
\]

这就是最小版本的训练。

```python
import torch

x = torch.tensor(2.0)
y = torch.tensor(5.0)
w = torch.tensor(0.0, requires_grad=True)

for step in range(5):
    loss = (w * x - y) ** 2
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
        w.grad.zero_()
    print(step, round(w.item(), 3), round(loss.item(), 3))
```

## 常见误区

- 把函数看成符号操作，不关心输入输出 shape。
- 认为导数只是一张公式表，不理解变化率。
- 忘记链式法则里的中间变量。
- 认为 softmax 输出大就一定代表可信概率。
- 图像只看高低，不看斜率和饱和区域。

## 检查问题

1. 函数 \(f_\theta(x)\) 中 \(x\)、\(\theta\)、输出分别是什么？
2. 导数和普通斜率有什么关系？
3. 为什么 sigmoid 两端梯度小？
4. softmax 输入整体变大时，输出分布会发生什么？
5. 链式法则如何对应反向传播？

## 分层练习

| 层级 | 任务 |
|---|---|
| 基础 | 画出 \(x^2\)、\(e^x\)、\(\log x\)、sigmoid 的取值表 |
| 推导 | 手推 \(L=(wx-y)^2\) 对 \(w\) 的导数 |
| 实现 | 用 PyTorch 训练一个单参数线性模型 |
| 诊断 | 改变学习率，观察 loss 是否下降或发散 |
| 迁移 | 解释 attention softmax 为什么可能过尖 |

## 阶段产出

整理一页“函数到训练”笔记：函数、导数、链式法则、梯度下降、softmax 饱和。每个概念配一个公式和一段 10 行以内代码。

## 教材补充：从函数到模型模块

把模型拆开看，每一层都是函数：

\[
h^{(l+1)}=f_l(h^{(l)};\theta_l).
\]

Transformer block 也可以写成函数复合：

\[
X\rightarrow \operatorname{Attention}(X)\rightarrow \operatorname{FFN}(X)\rightarrow L.
\]

读代码时可以把每一行都问成三个问题：输入是什么、函数是什么、输出是什么。这样会比直接背模块名更稳。

| 模块 | 函数视角 | 输入 shape | 输出 shape |
|---|---|---|---|
| Embedding | 查表函数 | `(B,T)` | `(B,T,D)` |
| Linear | 仿射函数 | `(B,T,D)` | `(B,T,M)` |
| GELU | 逐元素非线性 | `(B,T,D)` | `(B,T,D)` |
| Softmax | 分数到权重 | `(B,H,T,T)` | `(B,H,T,T)` |
| LayerNorm | 归一化函数 | `(B,T,D)` | `(B,T,D)` |

## 手推任务：两层函数复合

设：

\[
h=wx+b,
\quad
L=(h-y)^2.
\]

按链式法则：

\[
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial h}\frac{\partial h}{\partial w}
=2(h-y)x.
\]

把每个量写成 shape：若 `x` 是标量，`w` 是标量，梯度也是标量；若 `x` 是向量，`w` 是矩阵，就进入矩阵微分章节。

## Notebook 实验：观察函数尺度

```python
import torch

x = torch.linspace(-5, 5, 21)
funcs = {
    'linear': x,
    'square': x ** 2,
    'exp': torch.exp(x),
    'log_shift': torch.log(x - x.min() + 1),
    'sigmoid': torch.sigmoid(x),
}
for name, y in funcs.items():
    print(name, 'min=', round(y.min().item(), 3), 'max=', round(y.max().item(), 3))
```

观察重点不是具体数值，而是不同函数对输入尺度的敏感程度。指数函数会迅速变大；对数函数会压缩大数；sigmoid 会把输出限制在 0 到 1。

## 论文阅读提示

读 Transformer 论文中的函数式写法时，常见模式有：

- \(f_\theta(x)\)：带参数的模型。
- \(g(f(x))\)：函数复合，对应多层网络。
- \(\phi(x)\)：特征映射，常见于核方法或表示学习。
- \(\sigma(x)\)：非线性或 sigmoid，需看上下文。
- \(\operatorname{softmax}(x/\tau)\)：带温度的归一化函数。

遇到新符号时，先不要猜含义，先找输入输出和 shape。

## 章节小测

1. 为什么线性函数的导数是常数？
2. 为什么指数函数容易造成数值 overflow？
3. sigmoid 饱和时，对反向传播有什么影响？
4. softmax 为什么既是函数，也是分布构造方式？
5. 一个 Transformer block 可以看成哪些函数的复合？

## 本章完成标准

- 能把任意一行模型代码解释成函数输入、函数规则、函数输出。
- 能说清线性、指数、对数、sigmoid、softmax 的图像直觉。
- 能手推一个一层线性模型的梯度。
- 能解释函数饱和与梯度变小的关系。

