---
tags: [machine-learning, training]
---

# Regularization

## Purpose

Control the **complexity** of the model, and so the generalization - a principled approach, related to Tikhonov theory.

> [!warning] Frequent misunderstanding
> Regularization is **not** a technique to control the stability of the training convergence (the [[Learning Curve]]). It controls the complexity of the model, measured by the VC-dim - which in a NN depends on the number of weights **and** on their values.

## L2 / Tikhonov

$$
\operatorname{Loss}(\boldsymbol{w})=\underbrace{\sum_p(d_p-o(\boldsymbol{x}_p))^2}_{\text{error / data term (MSE)}}+\underbrace{\lambda\lVert \boldsymbol{w}\rVert^2}_{\text{penalty}},
\qquad \lVert \boldsymbol{w}\rVert^2=\sum_i w_i^2 \text{ over all the weights}.
$$

- Effect: **weight decay** - add $2\lambda w$ to the gradient of each weight:

  $$
  w_{new}=w+\eta\,\Delta w-2\lambda w \quad (\text{the }2\text{ can be omitted}).
  $$

- Higher $\lambda$: simpler model, smaller weights, higher bias, lower variance. Lower $\lambda$: the opposite.
- Typical values: generally very low, e.g. $0.01$. **Always selected by [[Model Selection and Assessment|model selection]].**
- Applied to a linear model this is [[Ridge Regression]].
- Visually: with $\lambda=0$ all the weights are significantly different from zero; with $\lambda=0.01$ most of them are close to zero $\to$ lower model complexity.

> [!important] Error vs Loss
> Keep the **Loss** (data term + penalty, the training objective) separate from the **Error / Risk** (the data term alone), because the error is what you report in tables and plots. See [[Loss Functions]].

## Details

- The bias $w_0$ is often left out of the regularizer (including it makes the result depend on target shift/scaling), or given its own coefficient.
- Typically applied in the batch version. For on-line/mini-batch, to compare fairly with batch do **not** force the same $\lambda$: use $\lambda\times(mb/l)$. (If you select $\lambda$ by model selection this happens automatically.)

## Putting momentum and weight decay together

Moving from $E$ to the full Loss, and using $\eta$ **only** on the error term:

$$
\Delta w_{tu}=\eta\,\delta_t o_u-\lambda w_{tu}+\alpha\,\Delta w_{old,tu},
\qquad w_{new}=w+\Delta w .
$$

This way $\eta$ does not multiply $\lambda$, so the two hyperparameters stay independent and are easier to control.

To keep all **three** independent (i.e. to decide whether the $\lambda$ part enters the momentum memory or not - many toolboxes include it):

$$
\Delta w_{tu}=\eta\,\delta_t o_u+\alpha\,\Delta w_{old,tu},
\qquad
w_{new,tu}=w_{tu}+\Delta w_{tu}-\lambda w_{tu}.
$$

Divide by $l$ only the gradient of $E$, where the sum over the patterns is.

## Other controls

- **L1 (lasso):** penalizes $|w_j|$ and drives some weights **exactly to zero** (while allowing others to be large) $\to$ towards [[Feature Selection]]. Not differentiable, so it needs other approaches.
- **Elastic nets:** both L1 and L2.
- [[Early Stopping]] (spends validation data).
- Data augmentation: inputs with added noise (teaches the model to ignore irrelevant variations), oversampling for imbalanced data.
- Derived inputs: principal component regression, partial least squares.
- Architecture / number of hidden units: too few $\to$ [[Underfitting]], too many $\to$ [[Overfitting]] (can be pushed higher if $\lambda$ is doing its job).
- **Architectural** regularization: local connections and weight sharing cut the number of free parameters while keeping the connections - see [[CNN]] and [[RNN]].
- [[Dropout]] - randomly drop units during training: a different mechanism for the same job.
- [[Batch Normalization]] - regularizes through the noise of the mini-batch statistics.

> [!note] Frequent misunderstanding - ES or Regularization?
> ES is empirical and needs a VL set, sacrificing part of the data. Regularization in the Loss is principled: it makes the VL curve follow the TR curve, so ES is not (strictly) needed and you can stop at training convergence. You can use both - and ES will simply not fire if the VL error never increases.

## Related

- [[Structural Risk Minimization]] - why this lowers the bound on $R$.
- [[Bias-Variance Tradeoff]] - the $\lambda$ pictures.
- [[Training]], [[Hyperparameter Guidelines]].
