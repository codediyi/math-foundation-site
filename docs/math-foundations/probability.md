# 概率统计

> 概率统计帮助你判断模型中的变化到底是规律、噪声还是偶然。Transformer 训练里的初始化、dropout、mini-batch、采样、softmax 和注意力缩放都需要概率直觉。

## 学习目标

- 理解事件、随机变量、分布、期望、方差、协方差和条件概率。
- 能解释 softmax 权重为什么不等于绝对可信概率。
- 能推导 attention 中 \(1/\sqrt{d_k}\) 缩放因子的直觉。
- 能用代码模拟维度、batch size、随机种子对统计量的影响。
- 能区分单次实验结果和统计上更稳健的结论。

## 从抛硬币开始

定义随机变量：

\[
X=\begin{cases}1,&\text{正面}\\0,&\text{反面}\end{cases}
\]

公平硬币：

\[
P(X=1)=0.5,
\quad
P(X=0)=0.5.
\]

期望：

\[
\mathbb{E}[X]=1\cdot 0.5+0\cdot 0.5=0.5.
\]

这不是说每次结果都是 0.5，而是长期平均接近 0.5。深度学习实验同理：一次训练结果不代表稳定结论。

## 随机变量和分布

| 对象 | 直觉 | 深度学习例子 |
|---|---|---|
| 随机变量 | 结果不固定 | 初始化权重、采样 token、mini-batch loss |
| 分布 | 结果及概率 | softmax 输出、数据分布、噪声分布 |
| 样本 | 一次观察 | 一个 batch 的 loss |
| 估计量 | 用样本估计总体量 | 验证集准确率、平均 loss |

## 期望：长期平均

离散期望：

\[
\mathbb{E}[X]=\sum_x xP(X=x).
\]

训练目标通常是期望损失：

\[
\mathcal{L}(\theta)=\mathbb{E}_{(x,y)\sim p_{data}}[\ell(f_\theta(x),y)].
\]

mini-batch 只是估计它：

\[
\hat{\mathcal{L}}=\frac{1}{B}\sum_{i=1}^{B}\ell_i.
\]

## 方差：波动有多大

\[
\operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2].
\]

方差大说明结果波动大。训练中表现为：loss 曲线抖、不同 seed 差异大、梯度 norm 波动大。

## 协方差：两个量是否一起变化

\[
\operatorname{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])].
\]

协方差为正：两个量倾向于一起变大。协方差为负：一个变大时另一个倾向于变小。

在表示学习中，协方差可以观察特征维是否高度相关。高度相关可能意味着冗余。

## 条件概率

条件概率：

\[
P(Y\mid X)=\frac{P(X,Y)}{P(X)}.
\]

语言模型建模的是：

\[
P(x_t\mid x_{<t}).
\]

即给定前文后，下一个 token 的分布。

## softmax 是归一化权重

\[
p_i=\frac{\exp(z_i)}{\sum_j\exp(z_j)}.
\]

它输出非负且和为 1 的数，因此可以当作分布使用。但模型很自信不代表一定正确，softmax 输出未必校准。

```python
import torch

for scale in [1.0, 5.0, 20.0]:
    z = torch.tensor([1.0, 2.0, 3.0]) * scale
    p = torch.softmax(z, dim=0)
    print(scale, p, 'max=', p.max().item())
```

## attention 缩放因子的推导

假设 \(q_i,k_i\) 独立，均值 0，方差 1。

\[
q^\top k=\sum_{i=1}^{d_k}q_ik_i.
\]

每一项 \(q_ik_i\) 的均值约为 0，方差约为 1。若各项近似独立：

\[
\operatorname{Var}(q^\top k)\approx d_k.
\]

除以 \(\sqrt{d_k}\)：

\[
\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right)\approx 1.
\]

这让 logits 尺度更稳定，避免 softmax 一开始就过尖。

## 用代码验证缩放

```python
import torch

torch.manual_seed(0)
for d in [16, 64, 256, 1024]:
    q = torch.randn(20000, d)
    k = torch.randn(20000, d)
    dot = (q * k).sum(dim=-1)
    scaled = dot / (d ** 0.5)
    print(d, round(dot.std().item(), 2), round(scaled.std().item(), 2))
```

## batch size 与估计方差

样本均值方差：

\[
\operatorname{Var}(\bar{X})=\frac{\operatorname{Var}(X)}{B}.
\]

```python
import torch

torch.manual_seed(0)
for B in [4, 16, 64, 256]:
    means = []
    for _ in range(1000):
        means.append(torch.randn(B).mean())
    means = torch.stack(means)
    print(B, round(means.std().item(), 3))
```

batch 越大，均值估计越稳定。但大 batch 也可能改变优化动态。

## 置信区间和多 seed

若只跑一次实验，指标提升可能来自随机波动。更稳妥的做法：

- 固定数据划分和评估脚本。
- 跑多个随机种子。
- 报告平均值和标准差。
- 对小提升保持谨慎。

## Transformer 对应关系

| 概率概念 | Transformer 中的对应 |
|---|---|
| 随机变量 | 初始化、dropout、采样 token、mini-batch loss |
| 期望 | 训练目标、泛化误差 |
| 方差 | logits 尺度、梯度噪声、实验波动 |
| 协方差 | 特征相关性、表示冗余 |
| 条件概率 | 语言模型 \(P(x_t\mid x_{<t})\) |
| softmax 分布 | attention 权重、词表预测分布 |
| 多 seed | 方法稳定性验证 |

## 常见误区

- 把期望理解为每次结果。
- 单次实验提升就认为方法有效。
- 忽略随机种子和置信区间。
- 把 softmax 输出当成校准概率。
- 推导 attention 缩放时忘记独立性假设。
- 只看平均值，不看方差和极端值。

## 检查问题

1. 随机变量和样本有什么区别？
2. 期望损失和 mini-batch loss 有什么区别？
3. 方差大在训练曲线上会表现为什么？
4. 为什么 \(q^\top k\) 的方差会随 \(d_k\) 增大？
5. softmax 很尖时，梯度和信息路由可能发生什么变化？
6. 为什么论文实验通常需要多个 seed？

## 分层练习

| 层级 | 任务 |
|---|---|
| 基础 | 模拟抛硬币 10/100/10000 次，观察均值 |
| 推导 | 推导 \(q^\top k\) 方差，并写出假设 |
| 实现 | 比较 `softmax(z)` 和 `softmax(10*z)` |
| 诊断 | 模拟不同 batch size 下均值估计标准差 |
| 迁移 | 给一个实验结果表补充 mean/std 汇报方式 |

## 阶段产出

写一页“概率统计到 Transformer 映射表”，解释 attention 缩放、softmax 饱和、mini-batch 噪声、随机种子、多次实验平均。每个条目都配一个最小代码观察。

## 教材补充：概率与校准

softmax 输出满足和为 1，但不代表模型概率一定校准。校准关注的是：模型说 80% 置信的样本，长期来看是否真的约 80% 正确。

过度自信是大模型中常见现象。低 loss、低 perplexity 不一定等于概率校准好。

| 现象 | 可能解释 |
|---|---|
| softmax 最大值很高但经常错 | 过度自信 |
| top-1 准确率高但概率不可信 | 排序能力强，校准差 |
| temperature scaling 后校准改善 | logits 尺度过大 |

## 手推任务：Bernoulli 方差

若 \(X\sim\operatorname{Bernoulli}(p)\)，则：

\[
\mathbb{E}[X]=p.
\]

因为 \(X^2=X\)，所以：

\[
\mathbb{E}[X^2]=p.
\]

方差：

\[
\operatorname{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2=p-p^2=p(1-p).
\]

当 \(p=0.5\) 时方差最大；当 \(p\) 接近 0 或 1 时方差小。

## Notebook 实验：多 seed 稳定性

```python
import torch

results = []
for seed in range(5):
    torch.manual_seed(seed)
    metric = 0.8 + 0.02 * torch.randn(()).item()
    results.append(metric)

x = torch.tensor(results)
print('mean=', x.mean().item())
print('std=', x.std(unbiased=True).item())
```

真实实验中，把一个方法的单次最好结果拿来比较，风险很高。至少要报告均值、标准差和设置。

## 条件期望与预测

监督学习可以理解为估计条件期望或条件分布。对于回归，平方损失下最优预测是：

\[
f^*(x)=\mathbb{E}[Y\mid X=x].
\]

对于语言模型，我们不是只预测一个均值，而是预测下一个 token 的条件分布：

\[
p(x_t\mid x_{<t}).
\]

这就是概率建模和普通回归的差别。

## 论文阅读提示

读实验部分时，概率统计帮助你判断结论是否稳：

| 论文表述 | 应追问 |
|---|---|
| improves by 0.2 | 是否超过随机波动 |
| average over seeds | seed 数是多少 |
| significant | 用了什么检验或置信区间 |
| sampling temperature | 温度如何影响输出分布 |
| variance reduction | 降低的是梯度方差还是评估方差 |

## 章节小测

1. 为什么 Bernoulli 方差是 \(p(1-p)\)？
2. softmax 输出为什么不等于校准概率？
3. batch size 增大为什么会让 loss 曲线更平滑？
4. 多 seed 平均能解决什么问题，不能解决什么问题？
5. attention 缩放推导依赖哪些假设？

## 本章完成标准

- 能手推 Bernoulli 期望和方差。
- 能解释 attention 缩放因子的概率直觉。
- 能写代码模拟 batch size 与估计方差。
- 能读实验表时主动关注随机性和置信度。

## 补充实验：校准分桶

校准可以用分桶直觉理解。把模型预测概率按置信度分成几组，观察每组真实正确率是否接近预测置信度。

```python
import torch

conf = torch.tensor([0.55, 0.62, 0.73, 0.81, 0.93])
correct = torch.tensor([1, 0, 1, 1, 0]).float()
print('avg confidence:', conf.mean().item())
print('avg accuracy:', correct.mean().item())
```

如果平均置信度远高于平均准确率，模型可能过度自信。真实项目中会用更多样本和更细分桶计算 ECE，但入门阶段先理解“置信度要和真实频率对齐”。

## 补充推导：样本均值为什么更稳定

若 \(X_1,\ldots,X_B\) 独立同分布，方差都是 \(\sigma^2\)，样本均值：

\[
\bar{X}=\frac{1}{B}\sum_{i=1}^BX_i.
\]

因为独立变量方差可加：

\[
\operatorname{Var}(\bar{X})
=\frac{1}{B^2}\sum_{i=1}^B\operatorname{Var}(X_i)
=\frac{B\sigma^2}{B^2}
=\frac{\sigma^2}{B}.
\]

这解释了 batch size 增大时 loss 曲线更平滑的统计原因。

## 最终复盘模板

```markdown
# 概率统计复盘

## 一个随机变量例子
## 一个期望和方差推导
## 一个 Transformer 概率对象
## 一个代码模拟结果
## 一个实验结论的统计风险
```

完成这份复盘后，再进入信息论会更自然，因为信息论中的熵、交叉熵和 KL 都建立在分布概念上。

