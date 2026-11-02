---
tags: [machine-learning, models]
---

# Ridge Regression

[[Linear Models]] + Tikhonov (L2) penalty:

$$
\operatorname{Loss}(\boldsymbol{w})=\sum_{p=1}^{l}(y_p-\boldsymbol{x}_p^T\boldsymbol{w})^2+\lambda\lVert \boldsymbol{w}\rVert^2 ,
\qquad \lVert \boldsymbol{w}\rVert^2=\sum_j w_j^2 .
$$

$\lambda$ is a **hyperparameter**, chosen in the [[Model Selection and Assessment|model selection]] phase.

## Solutions

- Direct: $\boldsymbol{w}=(\boldsymbol{X}^T\boldsymbol{X}+\lambda \boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y}$ - always invertible.
- Gradient: differentiate the two terms separately, using $\eta$ only on the error term:

  $$
  \boldsymbol{w}_{new}=\boldsymbol{w}+\eta\,\Delta \boldsymbol{w}-2\lambda \boldsymbol{w} .
  $$

  This is **weight decay**.

## Trade-off

- Small $\lambda$: focus on the data term, complex model (high weight norm) $\to$ [[Overfitting]].
- High $\lambda$: focus on the penalty, data error grows $\to$ [[Underfitting]].
- $\lambda$ rules the trade-off with a single number - easy to implement, generally applicable, and it directly lowers the effective VC-dim (see [[Structural Risk Minimization]]).

The intercept $w_0$ is typically left out of the penalty (otherwise the result depends on target shift/scaling), or given its own coefficient.

Same idea for NN: see [[Regularization]].
