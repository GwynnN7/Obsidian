---
tags: [machine-learning, models]
---

# Ridge Regression

[[Linear Models]] + Tikhonov (L2) penalty:

$$
\operatorname{Loss}(w)=\sum_{p=1}^{l}(y_p-x_p^Tw)^2+\lambda\lVert w\rVert^2 ,
\qquad \lVert w\rVert^2=\sum_j w_j^2 .
$$

$\lambda$ is a **hyperparameter**, chosen in the [[Model Selection and Assessment|model selection]] phase.

## Solutions

- Direct: $w=(X^TX+\lambda I)^{-1}X^Ty$ - always invertible.
- Gradient: differentiate the two terms separately, using $\eta$ only on the error term:

  $$
  w_{new}=w+\eta\,\Delta w-2\lambda w .
  $$

  This is **weight decay**.

## Trade-off

- Small $\lambda$: focus on the data term, complex model (high weight norm) $\to$ [[Overfitting]].
- High $\lambda$: focus on the penalty, data error grows $\to$ [[Underfitting]].
- $\lambda$ rules the trade-off with a single number - easy to implement, generally applicable, and it directly lowers the effective VC-dim (see [[Structural Risk Minimization]]).

The intercept $w_0$ is typically left out of the penalty (otherwise the result depends on target shift/scaling), or given its own coefficient.

Same idea for NN: see [[Regularization]].
