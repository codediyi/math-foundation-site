# 优化与数值稳定

> 优化回答“参数如何更新”，数值稳定回答“这个更新和计算是否可靠”。很多 Transformer 问题不是模型表达力不足，而是学习率、梯度尺度、归一化、softmax overflow、mixed precision 等细节失控。

## 学习目标

- 理解 GD、SGD、momentum、Adam、weight decay、warmup、gradient clipping。
- 能解释学习率过大、梯度爆炸、softmax overflow、NaN 的常见来源。
- 掌握稳定 softmax、log-sum-exp、cross entropy 的实现直觉。
- 能记录 loss、grad norm、parameter norm，并用它们诊断训练问题。
- 能把优化稳定性连接到 Transformer 的 Pre-LN、残差、warmup 和 mixed precision。

## 从一个最小例子开始

优化的目标是让损失变小。考虑：

\[
L(w)=(w-3)^2.
\]

梯度下降：

\[
w_{t+1}=w_t-\eta\nabla L(w_t).
\]

```python
w = 0.0
lr = 0.1
for step in range(8):
    loss = (w - 3) ** 2
    grad = 2 * (w - 3)
    w -= lr * grad
    print(step, round(w, 4), round(loss, 4))
```

如果学习率合适，loss 下降；如果学习率太大，参数会震荡甚至发散。

## GD 与 SGD

全量梯度下降使用全部数据：

\[
\nabla L(\theta)=\frac{1}{N}\sum_{i=1}^N\nabla \ell_i(\theta).
\]

SGD 或 mini-batch SGD 使用一个 batch 估计：

\[
\hat{g}=\frac{1}{B}\sum_{i\in\mathcal{B}}\nabla \ell_i(\theta).
\]

| 方法 | 优点 | 问题 |
|---|---|---|
| GD | 梯度稳定 | 大数据上太慢 |
| SGD | 计算便宜，有噪声探索 | 曲线抖动大 |
| Mini-batch SGD | 工程上折中 | batch size 会影响优化动态 |

## Momentum：让更新有惯性

Momentum 维护速度：

\[
v_t=\beta v_{t-1}+g_t,
\quad
\theta_t=\theta_{t-1}-\eta v_t.
\]

直觉：如果多个 step 的梯度方向一致，就加速；如果来回震荡，就部分抵消。

## Adam：自适应缩放梯度

Adam 维护一阶矩和二阶矩：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

更新大致为：

\[
\theta_t=\theta_{t-1}-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.
\]

直觉：梯度长期很大的维度会被缩小，梯度长期较小的维度会相对放大。但 Adam 不是万能的，学习率、warmup、weight decay 仍然很重要。

## Weight decay 与 L2 正则

AdamW 中常用 decoupled weight decay：

\[
\theta \leftarrow \theta-\eta\lambda\theta.
\]

直觉：每一步把参数往 0 拉一点，限制参数范数。它不是简单的“防止过拟合”按钮，也会改变优化路径。

## Warmup：训练早期慢慢加大学习率

Transformer 训练常用 warmup。原因包括：

- 初始参数和 Adam moments 还不稳定。
- 早期 logits、激活、梯度尺度可能变化剧烈。
- 直接使用大学习率容易让模型跳到不稳定区域。

线性 warmup：

```python
def lr_schedule(step, base_lr=3e-4, warmup=1000):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr
```

## 梯度裁剪

梯度范数过大时，更新可能异常。global norm clipping：

\[
g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|}\right).
\]

```python
import torch

params = [torch.randn(10, requires_grad=True)]
loss = (params[0] ** 2).sum()
loss.backward()
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
print(params[0].grad.norm())
```

裁剪不是为了让所有梯度都小，而是防止少数异常 batch 造成巨大更新。

## 数值稳定：overflow 和 underflow

指数函数增长很快：

```python
import torch
print(torch.exp(torch.tensor(100.0)))
```

在某些 dtype 下会 overflow。softmax 直接对大 logits 做 `exp` 会出现 inf。

稳定 log-sum-exp：

\[
\log\sum_i e^{z_i}=m+\log\sum_i e^{z_i-m},\quad m=\max_i z_i.
\]

因为减去最大值不会改变 softmax 比例。

## 稳定 softmax

```python
import torch

def stable_softmax(x, dim=-1):
    z = x - x.max(dim=dim, keepdim=True).values
    e = z.exp()
    return e / e.sum(dim=dim, keepdim=True)

x = torch.tensor([[1000.0, 1001.0, 1002.0]])
print(stable_softmax(x))
print(torch.softmax(x, dim=-1))
```

PyTorch 内置 `softmax` 已做稳定处理，但自己写实现时必须记住这个技巧。

## 稳定 cross entropy

不要先 softmax 再 log：

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[1000.0, 1001.0, 1002.0]])
target = torch.tensor([2])
loss = F.cross_entropy(logits, target)
print(loss)
```

`cross_entropy` 内部使用稳定的 log-softmax。手写 `torch.log(torch.softmax(logits))` 更容易数值出错。

## mixed precision 常见问题

混合精度能加速训练、节省显存，但也更容易遇到：

| 问题 | 现象 | 常见处理 |
|---|---|---|
| overflow | loss/grad 变 NaN 或 Inf | loss scaling、clip、降低 lr |
| underflow | 小梯度变 0 | 使用 bf16/fp32 master weights |
| softmax 不稳定 | attention 出 NaN | stable softmax、mask 检查 |
| LayerNorm eps 太小 | 方差接近 0 时异常 | 合理设置 eps |

## 训练诊断指标

建议每次训练记录：

| 指标 | 用途 |
|---|---|
| train loss | 是否下降、是否震荡 |
| eval loss | 是否泛化 |
| grad norm | 是否爆炸或异常变 0 |
| param norm | 参数是否持续变大 |
| learning rate | 与 loss 变化对齐分析 |
| max logits | 判断 softmax 是否过尖 |
| NaN/Inf count | 及时定位数值错误 |

## Transformer 对应关系

| 优化/稳定概念 | Transformer 中的位置 |
|---|---|
| AdamW | 大多数 Transformer 训练默认优化器 |
| warmup | 训练早期更新尺度控制 |
| gradient clipping | 防止异常 batch 造成大更新 |
| stable softmax | attention 和 vocab softmax |
| log-sum-exp | cross entropy、log likelihood |
| Pre-LN | 深层训练稳定性 |
| mixed precision | 大模型训练速度和显存 |

## 逐步例题：学习率导致发散

```python
for lr in [0.05, 0.5, 1.1]:
    w = 5.0
    losses = []
    for _ in range(8):
        loss = w ** 2
        grad = 2 * w
        w -= lr * grad
        losses.append(round(loss, 3))
    print('lr=', lr, losses)
```

观察：小学习率稳定但慢；中等学习率快；过大学习率发散。

## 常见误区

- 认为 Adam 会自动解决所有训练不稳定。
- 只调学习率，不看 grad norm 和 logits 尺度。
- 手写 `softmax` 忘记减最大值。
- 把 NaN 当成随机现象，不记录具体 step 和 batch。
- mixed precision 下不检查 overflow/underflow。
- weight decay、dropout、label smoothing 混在一起调，无法判断原因。

## 检查问题

1. SGD 和 GD 的梯度有什么区别？
2. Adam 为什么要维护一阶矩和二阶矩？
3. warmup 解决的是哪类早期训练问题？
4. gradient clipping 改变的是梯度方向还是尺度？
5. log-sum-exp 为什么能避免 overflow？
6. 训练出现 NaN 时，你会先检查哪些指标？

## 分层练习

| 层级 | 任务 |
|---|---|
| 基础 | 在 \(w^2\) 上比较 3 个学习率 |
| 推导 | 推导 stable softmax 减最大值不改变结果 |
| 实现 | 写 stable softmax 和 stable cross entropy 对比 |
| 诊断 | 记录一个小模型的 loss、lr、grad norm |
| 迁移 | 分析 Pre-LN 为什么比 Post-LN 更稳定 |

## 阶段产出

写一页“训练稳定性检查表”，包含学习率、warmup、AdamW、weight decay、grad norm、stable softmax、NaN/Inf、mixed precision。每项都要写“可能症状”和“排查方法”。

## 教材补充：把训练问题分成三类

训练异常不要只说“模型不收敛”。先分成三类：

| 类型 | 典型现象 | 优先检查 |
|---|---|---|
| 优化问题 | loss 不降、震荡、发散 | lr、warmup、optimizer、grad norm |
| 数值问题 | NaN、Inf、overflow | logits、softmax、dtype、mask |
| 泛化问题 | train 降，eval 不降 | 数据、正则、过拟合、评估协议 |

这三类问题处理方法不同。优化问题不一定靠换模型解决；数值问题也不该靠盲目调参掩盖。

## 手推任务：gradient clipping 比例

若原始梯度范数为 \(\|g\|=10\)，阈值 \(c=2\)，裁剪后：

\[
g'=g\cdot\frac{2}{10}=0.2g.
\]

方向不变，尺度变小。若 \(\|g\|<c\)，则不改变梯度。

## Notebook 实验：检测 NaN 和 Inf

```python
import torch

x = torch.tensor([1.0, float('inf'), float('nan')])
print(torch.isfinite(x))
print('has bad value:', (~torch.isfinite(x)).any().item())
```

训练中建议对 loss、grad norm、logits max 做类似检查，尽早定位异常 step。

## 学习率日志模板

| step | lr | train loss | grad norm | param norm | max logit | note |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 1e-5 | 8.2 | 0.9 | 120 | 15 | warmup |
| 1000 | 3e-4 | 4.1 | 2.4 | 126 | 28 | stable |
| 1500 | 3e-4 | NaN | Inf | 130 | Inf | check mask/softmax |

如果没有日志，只凭最终指标很难定位训练问题。

## 论文阅读提示

优化相关论文常见关键词：

| 关键词 | 应关注 |
|---|---|
| warmup steps | 前期 lr 增长策略 |
| cosine decay | 后期 lr 衰减方式 |
| AdamW betas | 一阶/二阶矩平滑强度 |
| gradient clipping | 是否限制异常更新 |
| bf16/fp16 | 是否涉及 loss scaling |
| Pre-LN/Post-LN | 梯度路径差异 |
| stability | 是否给出 grad norm 或 loss 曲线证据 |

## 章节小测

1. AdamW 和 Adam 加 L2 正则有什么概念差别？
2. gradient clipping 为什么不等于简单减小学习率？
3. `log(softmax(x))` 为什么不如 `log_softmax(x)` 稳定？
4. mixed precision 下 NaN 的常见来源有哪些？
5. 训练曲线震荡时，你会先看哪些日志？

## 本章完成标准

- 能实现 stable softmax 和解释 log-sum-exp。
- 能说清 Adam、warmup、weight decay、gradient clipping 的作用。
- 能设计一张训练稳定性日志表。
- 能把 NaN/Inf 排查分解到 logits、mask、dtype、梯度四类线索。

## 补充实验：更新范数

除了梯度范数，还可以记录参数更新范数：

\[
\frac{\|\Delta\theta\|}{\|\theta\|}.
\]

如果这个比例突然变大，说明某一步更新相对参数尺度过猛，可能导致发散。

```python
import torch

theta = torch.randn(100)
grad = torch.randn(100)
lr = 1e-3
update = -lr * grad
ratio = update.norm() / theta.norm()
print(ratio.item())
```

在大模型训练中，单看 loss 往往太晚；更新范数、grad norm、max logits 能更早暴露问题。

## 最终复盘模板

```markdown
# 优化与数值稳定复盘

## 本次训练使用的 optimizer/lr/warmup
## loss 曲线是否稳定
## grad norm 和 update norm 是否异常
## 是否出现 NaN/Inf
## softmax、mask、dtype 排查记录
```

