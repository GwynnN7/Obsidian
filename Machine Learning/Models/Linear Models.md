---
tags: [machine-learning, models]
---

# Linear Models

Still a restricted $H$, but now a **continuous** one. Baseline for everything else: first question on any task is "is it linear?".

$$
h_{\boldsymbol{w}}(\boldsymbol{x})=\boldsymbol{w}^T\boldsymbol{x}=w_0+\sum_{i=1}^{n}w_ix_i ,
$$

with the convention $x_0=1$ so that $w_0$ (intercept / threshold / bias / offset) is inside the product.

> [!note]
> "Linear" refers to how the parameters $\boldsymbol{w}$ enter the equation, not to the shape of the curve in $\boldsymbol{x}$. That is why [[Linear Basis Expansion]] is still a linear model.

## LMS / least squares

Given $l$ examples $(\boldsymbol{x}_p,y_p)$, find $\boldsymbol{w}$ minimizing the residual sum of squares:

$$
E(\boldsymbol{w})=\sum_{p=1}^{l}(y_p-\boldsymbol{x}_p^T\boldsymbol{w})^2=\lVert \boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}\rVert^2 .
$$

Divide by $l$ to get the mean (LMS). This is a quadratic, convex function $\to$ a minimum always exists (not necessarily unique), no local minima.

## Two learning algorithms

### 1. Direct: [[Normal Equation]]

Setting the gradient to zero,

$$
\frac{\partial E(\boldsymbol{w})}{\partial w_j}=-2\sum_{p=1}^{l}(y_p-\boldsymbol{x}_p^T\boldsymbol{w})\,x_{p,j}=0
\quad\Longrightarrow\quad
(\boldsymbol{X}^T\boldsymbol{X})\boldsymbol{w}=\boldsymbol{X}^T\boldsymbol{y} .
$$

- If $\boldsymbol{X}^T\boldsymbol{X}$ is non singular: $\boldsymbol{w}=(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}=\boldsymbol{X}^{+}\boldsymbol{y}$ with $\boldsymbol{X}^{+}$ the Moore-Penrose pseudoinverse.
- Otherwise infinite solutions: pick the minimum-norm $\boldsymbol{w}$, computed with SVD.
- Cost is cubic in the matrix dimension.

### 2. Iterative: [[Gradient Descent]]

$$
\Delta w_j=2\sum_{p=1}^{l}\underbrace{(y_p-\boldsymbol{x}_p^T\boldsymbol{w})}_{\delta_p}x_{p,j},
\qquad \boldsymbol{w}_{new}=\boldsymbol{w}+\eta\,\Delta \boldsymbol{w} .
$$

Preferred, because it is:

- cheaper than the direct solution,
- compatible with [[Regularization]],
- able to stop before the minimum on noisy data,
- **applicable to non-linear models too** - this is the road to [[Neural Networks]].

This is the [[Delta Rule]] / Widrow-Hoff error-correction rule: no error $\to$ no correction; output too low with positive input $\to$ increase $w_j$.

## Classification (LTU)

Threshold the linear score:

$$
h(\boldsymbol{x})=\begin{cases}1 & \text{if } \boldsymbol{w}^T\boldsymbol{x}+w_0\ge0\\ 0 & \text{otherwise}\end{cases}
\qquad\text{or}\qquad
h(\boldsymbol{x})=\operatorname{sign}(\boldsymbol{w}^T\boldsymbol{x}+w_0).
$$

- $\boldsymbol{w}^T\boldsymbol{x}+w_0=0$ is the **decision boundary**, a hyperplane. Saying $\boldsymbol{w}^T\boldsymbol{x}\ge-w_0$ shows $-w_0$ as the activation threshold.
- $\boldsymbol{w}$ is **orthogonal** to the hyperplane: for $\boldsymbol{x}_a,\boldsymbol{x}_b$ on it, $\boldsymbol{w}^T(\boldsymbol{x}_a-\boldsymbol{x}_b)=0$.
- If $w_0=0$ the hyperplane goes through the origin.
- **Scaling freedom:** multiplying $\boldsymbol{w}$ by any $K>0$ gives the same boundary.
- If a separating hyperplane exists there are many of them - hence many different learning algorithms.

For training we still use the squared error on $w^Tx$ (not on $h(x)$), so the loss stays differentiable. Minimizing MSE also drives the classification error down, but:

> [!warning]
> LMS does **not** minimize the number of misclassifications: it changes the weights also for correctly classified patterns, so it can misclassify even on linearly separable problems. That is the difference with the [[Perceptron]] learning algorithm.

## Limits

- Exact solutions only for linearly separable sets. XOR is the classic counterexample (4 points, no separating line).
- Too rigid for anything with real structure.
- Way out: [[Linear Basis Expansion]], or adaptive basis functions = [[Neural Networks]].

## Complexity control

- [[Ridge Regression]] / Tikhonov: $\operatorname{Loss}(\boldsymbol{w})=E(\boldsymbol{w})+\lambda\lVert \boldsymbol{w}\rVert^2$.
- L1 (lasso), elastic nets - see [[Regularization]].
- More basis functions $\to$ more expressivity $\to$ more overfitting risk, plus [[Curse of Dimensionality]].
- Data augmentation, derived inputs (principal component regression, partial least squares).

## Inductive bias

- **Language bias:** $H$ is the set of linear functions - very restrictive.
- **Search bias:** ordered search guided by least-squares minimization.

Related: [[Classification]], [[Regression]], [[k-NN]], [[Perceptron]].
