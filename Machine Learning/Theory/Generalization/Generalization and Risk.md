---
tags: [machine-learning, theory]
---

# Generalization and Risk

What "good" means in ML: low loss on data the model has **never seen**, not low error on the training set.

## The result

Two quantities, and the entire subject lives in the gap between them.

$$
\textbf{true (expected) risk:}\qquad R(h)=\mathbb{E}_Z[L(z,h)]=\int_Z L(z,h)\,p(z)\,dz
$$

$$
\textbf{empirical risk:}\qquad R_{emp}(h,D_r)=\frac1r\sum_{p=1}^{r}L(z_p,h)
$$

What we would like is $h^{*}=\arg\min_{h\in H}R(h)$. What training actually does is **Empirical Risk Minimization**: minimize $R_{emp}(h,D_{TR})$ and hope.

| symbol | what it is |
|---|---|
| $L(h(x),d)$ | the **inner** loss on one pattern - see [[Loss Functions]] |
| $z=(x,d)$ | one example; $Z_1,\dots,Z_l$ drawn **i.i.d.** from an unknown $p(z)$ |
| $D_r$ | one particular finite set of $r$ of them |
| $R(h)$ | average loss over the **whole distribution** - unknowable, $p(z)$ is unknown |
| $R_{emp}(h,D_r)$ | average loss on the $r$ examples you have - computable |

> [!important]
> $R_{emp}$ is what you can compute; $R$ is what you want. **They are not the same number.** Whether the first says anything about the second depends on the amount of data and the capacity of $H$ - that is exactly the question [[Statistical Learning Theory]] answers.

## Two phases

- **Training / fitting:** learn the parameters from known data.
- **Prediction / inference (deployment):** apply $h$ to new examples and compare with targets it has never seen.

> Performance in ML = generalization / predictive accuracy.

## The setting in detail

- Classification: $Z=(X,Y)\in\mathbb{R}^n\times\{-1,+1\}$.
- Regression: $Z=(X,Y)\in\mathbb{R}^n\times\mathbb{R}$.
- $R(h)$ is an integral against $p(z)$, which we cannot evaluate: the distribution is unknown and we cannot access $L(z,h)$ for all $z$. $R_{emp}$ replaces the integral with a finite average - a **surrogate**, nothing more.

## Empirical Risk Minimization (ERM)

Training minimizes $R_{emp}(h,D_{TR})$ as a finite-data stand-in for $R(h)$. That is an *inductive principle*, not a theorem.

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
