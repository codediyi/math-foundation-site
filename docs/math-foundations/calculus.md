# 微积分复习

## 学习目标

掌握极限、连续、导数、Taylor 展开和链式法则的研究用法。重点不是刷计算题，而是理解局部线性化、误差项和梯度下降为什么能工作。

## 为什么要学

深度学习训练依赖“局部近似”。当参数从 \(\theta\) 更新到 \(\theta+\Delta\theta\) 时，我们默认损失函数可以被一阶或二阶展开近似：

\[
L(\theta+\Delta\theta) \approx L(\theta) + \nabla L(\theta)^\top \Delta\theta.
\]

如果不理解 Taylor 展开，就很难理解学习率、smoothness、梯度爆炸和二阶优化。

## 核心概念

- 极限与连续：判断函数在小扰动下是否稳定。
- 导数与方向导数：衡量沿某个方向的局部变化率。
- Taylor 展开：把非线性函数局部写成多项式近似。
- Lipschitz 与 smoothness：约束函数变化速度和梯度变化速度。

## Transformer 对应关系

| 微积分概念 | Transformer 中的对象 |
|---|---|
| 链式法则 | 反向传播穿过 attention、FFN、LayerNorm |
| Taylor 展开 | 优化步长、局部线性化、训练稳定性 |
| Lipschitz | 残差块、归一化、梯度传播上界 |
| 非光滑点 | ReLU、mask、clip、top-k 路由 |

## 最小练习

证明 softmax 的平移不变性：

\[
\operatorname{softmax}(z+c\mathbf{1}) = \operatorname{softmax}(z).
\]

这一步是稳定 softmax 的数学基础。

## 关键代码块

```python
import numpy as np

def numerical_derivative(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

f = lambda x: x**2 + 3 * x
print(numerical_derivative(f, 2.0))  # close to 7
```

## 阶段产出

写一页笔记解释：为什么梯度下降是用局部线性近似选择下降方向，而不是一次性求出全局最优。

