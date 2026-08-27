---
tags: [machine-learning, theory]
---

# Generalization and Risk

## Two phases

- **Training / fitting:** learn the parameters from known data.
- **Prediction / inference (deployment):** apply $h$ to new examples and compare with targets it has never seen.

Goal: low loss on unseen data - **not** low training error.

> Performance in ML = generalization/predictive accuracy.

## Loss and risk

- Inner loss on one pattern: $L(h(x),d)$ - see [[Loss Functions]].
- Data: $Z_1,\dots,Z_l$ i.i.d. from an unknown $p(z)$; $D_r=\{z_1,\dots,z_r\}$ is one particular instance.
  - Classification: $Z=(X,Y)\in\mathbb{R}^n\times\{-1,+1\}$.
  - Regression: $Z=(X,Y)\in\mathbb{R}^n\times\mathbb{R}$.
- **True / expected risk:**

  $$
  R(h)=\mathbb{E}_Z[L(z,h)]=\int_Z L(z,h)\,p(z)\,dz.
  $$

  We would like $h^*=\arg\min_{h\in H}R(h)$, but $p(z)$ is unknown and we cannot access all $L(z,h)$.
- **Empirical risk** on a finite set:

  $$
  R_{emp}(h,D_r)=\frac1r\sum_{p=1}^{r}L(z_p,h).
  $$

## Empirical Risk Minimization (ERM)

Training minimizes $R_{emp}(h,D_{TR})$, a finite-data surrogate for $R(h)$.

The whole question: **can we use $R_{emp}$ to approximate $R$?** Answer: only under conditions on the number of data and on the capacity of $H$ - see [[Statistical Learning Theory]].

## [[Overfitting]]

A hypothesis can fit the training data better and still be worse on new data. There is $h'$ with

$$
E' > E, \qquad R' < R.
$$

> [!warning]
> Training error is not a reliable estimate of generalization error - and neither is validation error a reliable estimate of test error. See [[Model Selection and Assessment]].

> [!note]
> Everything here assumes the data are i.i.d. from a fixed distribution. When that stops being true you get **concept drift** - see [[Reasoning over Time]].

## Related

- [[Bias-Variance Tradeoff]]
- [[Regularization]]
- [[Validation]]
