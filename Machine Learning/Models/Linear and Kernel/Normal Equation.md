---
tags: [machine-learning, models]
---

# Normal Equation

Closed-form least-squares condition, obtained by setting $\partial E(\boldsymbol{w})/\partial \boldsymbol{w}=\boldsymbol{0}$:

$$
(\boldsymbol{X}^T\boldsymbol{X})\boldsymbol{w}=\boldsymbol{X}^T\boldsymbol{y} .
$$

- Non-singular $\boldsymbol{X}^T\boldsymbol{X}$: $\boldsymbol{w}=(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}=\boldsymbol{X}^{+}\boldsymbol{y}$, with $\boldsymbol{X}^{+}$ the Moore-Penrose pseudoinverse.
- Otherwise: infinite solutions satisfying the equation $\to$ choose the minimum-norm $\boldsymbol{w}$.
- **SVD** is the practical tool: $\boldsymbol{X}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T \Rightarrow \boldsymbol{X}^{+}=\boldsymbol{V}\boldsymbol{\Sigma}^{+}\boldsymbol{U}^T$ (replace every nonzero entry of $\Sigma$ by its reciprocal), then $\boldsymbol{w}=\boldsymbol{X}^{+}\boldsymbol{y}$ directly gives the minimal-norm least-squares solution.
- With [[Ridge Regression]]: $\boldsymbol{w}=(\boldsymbol{X}^T\boldsymbol{X}+\lambda \boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y}$ - and this matrix is **always** invertible.

Cost is cubic in the matrix dimension, which is why [[Gradient Descent]] is usually preferred.

See [[Linear Models]].
