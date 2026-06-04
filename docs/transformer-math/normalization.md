# 残差与归一化

## 学习目标

理解残差连接、LayerNorm、RMSNorm、Pre-LN、Post-LN 和 DeepNorm 的训练稳定性直觉。

## 为什么要学

深层 Transformer 的训练不是简单堆层数。残差让梯度有近似恒等路径，归一化控制激活尺度，二者共同决定深层传播是否稳定。

## 核心概念

- 残差：\(x_{l+1}=x_l+f(x_l)\)。
- LayerNorm：按 hidden 维归一化均值和方差。
- RMSNorm：只按均方根缩放。
- Pre-LN：先归一化再进子层。
- Post-LN：子层和残差之后再归一化。

## 关键公式

残差块的 Jacobian：

\[
\frac{\partial x_{l+1}}{\partial x_l}=I+J_f(x_l).
\]

这个 \(I\) 是深层梯度传播的重要直觉来源。

## Transformer 对应关系

| 结构 | 稳定性含义 |
|---|---|
| Residual | 提供恒等梯度路径 |
| LayerNorm | 控制激活尺度 |
| Pre-LN | 改善深层训练早期稳定性 |
| DeepNorm | 调整残差尺度以支持更深网络 |

## 关键代码块

```python
import torch

def layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)
```

## 最小练习

手写 LayerNorm 前向，和 `torch.nn.LayerNorm` 在不含 affine 参数时比较输出。然后改变输入尺度，观察归一化前后的均值和方差。

## 阶段产出

写一页笔记解释：为什么 Pre-LN 往往更容易训练，但 Post-LN 有时在最终性能上仍有吸引力。

