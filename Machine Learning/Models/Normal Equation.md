---
tags: [machine-learning, models]
---

# Normal Equation

Closed-form least-squares condition, obtained by setting $\partial E(w)/\partial w=0$:

$$
(X^TX)w=X^Ty .
$$

- Non-singular $X^TX$: $w=(X^TX)^{-1}X^Ty=X^{+}y$, with $X^{+}$ the Moore-Penrose pseudoinverse.
- Otherwise: infinite solutions satisfying the equation $\to$ choose the minimum-norm $w$.
- **SVD** is the practical tool: $X=U\Sigma V^T \Rightarrow X^{+}=V\Sigma^{+}U^T$ (replace every nonzero entry of $\Sigma$ by its reciprocal), then $w=X^{+}y$ directly gives the minimal-norm least-squares solution.
- With [[Ridge Regression]]: $w=(X^TX+\lambda I)^{-1}X^Ty$ - and this matrix is **always** invertible.

Cost is cubic in the matrix dimension, which is why [[Gradient Descent]] is usually preferred.

See [[Linear Models]].
