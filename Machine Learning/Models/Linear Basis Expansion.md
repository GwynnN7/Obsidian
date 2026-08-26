---
tags: [machine-learning, models]
---

# Linear Basis Expansion

Augment the input with fixed nonlinear transformations $\phi_k:\mathbb{R}^n\to\mathbb{R}$, then fit a model that is still **linear in the parameters**:

$$
h_w(x)=\sum_{k=0}^{K}w_k\phi_k(x)
\qquad\text{(classification: } h(x)=f\big(\textstyle\sum_k w_k\phi_k(x)\big)\text{)}.
$$

Examples of $\phi$:

- polynomial: $\phi(x)=x_j^2$, $\phi(x)=x_jx_i$ - the 1-dim case is polynomial regression of degree $M$;
- nonlinear transform of a single input: $\log(x_j)$, $\sqrt{x_j}$;
- of multiple inputs: $\lVert x\rVert$;
- splines, ...

E.g. $h(x)=w_1x_1+w_2x_2+w_3\log(x_2)+w_4\log(x_3)+w_5(x_2x_3)+w_0$.

Typically $K>n$. The learning algorithm is unchanged (LMS, [[Normal Equation]] or [[Gradient Descent]]).

## Pros / cons

- **Pro:** much more expressive than plain [[Linear Models]].
- **Con:** with a large basis, overfitting comes fast $\to$ needs [[Regularization]]; and [[Curse of Dimensionality]].
- **Con:** the $\phi$ are fixed **before** seeing the training data.

That last point is the whole motivation for [[Neural Networks]], where the basis functions are **adaptive**:

$$
\phi_j(x,w)=f_j\Big(\sum_i w_{ji}x_i\Big).
$$

[[SVM]] is another form of LBE, with a **fixed** $\phi$ made modular by [[Kernel Methods]] - kernels compute the feature-space inner product implicitly, so "kernel" is not a synonym of LBE but a way of computing one. [[Randomized Neural Networks]] are a *randomized* one.

Why expanding helps at all: [[Cover's Theorem]].

See [[Linear Models]].
