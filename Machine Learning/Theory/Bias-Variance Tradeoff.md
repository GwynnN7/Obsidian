---
tags: [machine-learning, theory]
---

# Bias-Variance Tradeoff

A training set is only **one** possible realization from the universe of data: different training sets give different models. Bias-variance decomposes the error expected over those realizations.

Setting: regression, squared error loss, target $y$.

$$
\mathbb{E}[\text{error at } x]=\text{bias}^2+\text{variance}+\text{noise}.
$$

- **Bias:** $[\bar h(x)-f(x)]^2$, the systematic discrepancy between the true function and the model output averaged over training sets. High when $H$ is too small / too rigid.
- **Variance:** variability of $h(x)$ across different realizations of the training data. High when the model is too flexible.
- **Noise:** irreducible; the labels contain random error (e.g. more than one possible $d$ for the same $x$). Even the optimal solution can be wrong. Does not depend on the model.

> [!important]
> Bias and variance are computed **across possible training sets**, not from one training run.

## The lemma used in the derivation

Starting from $\bar Z=\mathbb{E}_P[Z]=\sum_i z_i P(z_i)$ and $\operatorname{Var}[Z]=\mathbb{E}[(Z-\bar Z)^2]$:

$$
\operatorname{Var}[Z]=\sum_i (z_i^2-2z_i\bar Z+\bar Z^2)P(z_i)=\mathbb{E}[Z^2]-2\bar Z\bar Z+\bar Z^2=\mathbb{E}[Z^2]-\bar Z^2 .
$$

Used in the form

$$
\mathbb{E}[Z^2]=\bar Z^2+\operatorname{Var}[Z],
$$

applied twice (once to the target with the noise, once to $h$ over the training sets) to split the expected squared error into the three terms.

## Capacity and regularization

Same experiment (25 data sets from a sinusoid, RBF network with LMS), varying $\lambda$ in [[Regularization]]:

| $\lambda$ | effect |
|---|---|
| high | penalty dominates, model too rigid: **high bias, low variance** $\to$ [[Underfitting]] |
| medium | the trade-off |
| low | strongly dependent on the specific training data: **low bias, high variance** $\to$ [[Overfitting]] |

With very low $\lambda$ the single fits are wild, but the **mean** $\bar h(x)$ is still close to the true function - which is exactly the idea behind [[Bagging]].

> [!note]
> In the classic $\lambda$ plot the complexity grows to the **left** (low $\lambda$), the reverse of our usual capacity plots.

## Why it matters for validation

The decomposition explains why estimating a model's performance is hard: a single train/test split can be lucky or unlucky. Hence repeated splits, [[Cross Validation]], and reporting mean $\pm$ std.

## Ensembles

- [[Ensemble Learning]] - combine several models.
- [[Bagging]] - mainly reduces **variance**.
- [[Boosting]] - mainly reduces **bias**.

## Related

- [[Statistical Learning Theory]] - same trade-off, analytical version.
- [[Regularization]], [[Model Selection and Assessment]], [[Feature Selection]].
