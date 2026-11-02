---
tags: [machine-learning, reference]
---

# Vector and Matrix Algebra

The small set of algebra every model in these notes runs on. Not a maths course - the operations that keep coming back, each with the place it is actually used.

Symbols and what is bold: [[Notation]].

## Conventions

- Vectors are **columns** by default: $\boldsymbol{x}\in\mathbb{R}^n$ is $n\times1$, and $\boldsymbol{x}^T$ is the row.
- Data matrix $\boldsymbol{X}$ is $l\times n$: **one row per pattern**, one column per feature. So row $p$ is $\boldsymbol{x}_p^T$.
- $p=1..l$ runs over patterns, $i,j=1..n$ over components.

## Sums

$$\sum_{i=1}^{n} a_i$$

- Linearity: $\sum_i(a_i+b_i)=\sum_i a_i+\sum_i b_i$, and constants come out, $\sum_i c\,a_i=c\sum_i a_i$.
- A constant summed: $\sum_{i=1}^{n}c=nc$. The case $\sum_i P(z_i)=1$ is what closes the variance lemma in [[Bias-Variance Tradeoff]].
- Finite double sums commute and can be split - used all over the [[SVM]] dual.

> [!warning] The index is the meaning
> $\sum_p$ runs over **patterns** (how many data), $\sum_i$ over **components** (how many features). Mixing them is the most common slip in a derivation: an error is averaged over $p$, a norm is summed over $i$.

## Dot product

$$\boldsymbol{a}^T\boldsymbol{b}=\sum_{i=1}^{n}a_ib_i \qquad\text{(a scalar)}$$

Also written $\boldsymbol{a}\cdot\boldsymbol{b}$, $\langle\boldsymbol{a},\boldsymbol{b}\rangle$, or just $\boldsymbol{a}\boldsymbol{b}$ when the context is clear. Symmetric and bilinear.

Geometrically:

$$\boldsymbol{a}^T\boldsymbol{b}=\lVert \boldsymbol{a}\rVert\,\lVert \boldsymbol{b}\rVert\cos\theta$$

- $=0$ $\to$ the two vectors are **orthogonal**.
- Its **sign** says which side of a hyperplane a point is on - the whole of [[Linear Models]] and [[Perceptron]].
- **Cauchy-Schwarz:** $|\boldsymbol{a}^T\boldsymbol{b}|\le\lVert \boldsymbol{a}\rVert\lVert \boldsymbol{b}\rVert$ (it follows from $|\cos\theta|\le1$). This is the upper bound in the [[Perceptron]] convergence proof.

## Norms

| Norm                                 | Definition                                                  |
| ------------------------------------ | ----------------------------------------------------------- |
| $\lVert \boldsymbol{x}\rVert$        | $\sqrt{\boldsymbol{x}^T\boldsymbol{x}}=\sqrt{\sum_i x_i^2}$ |
| $\lVert \boldsymbol{x}\rVert^2$      | $\boldsymbol{x}^T\boldsymbol{x}=\sum_i x_i^2$               |
| $\lVert \boldsymbol{x}\rVert_1$      | $\sum_i\lvert x_i\rvert$                                    |
| $\lVert \boldsymbol{x}\rVert_\infty$ | $\max_i\lvert x_i\rvert$                                    |

> [!important] Why the square is everywhere
> $\lVert \cdot\rVert^2$ avoids the square root, which is **not differentiable at 0** and produces an ugly gradient. The squared norm is smooth and its gradient is just $2\boldsymbol{x}$. And since $\sqrt{\cdot}$ is monotone on non-negatives, minimizing $\lVert \cdot\rVert$ and $\lVert \cdot\rVert^2$ give the **same** solution - so you may always drop the root.

L1 vs L2 in a penalty ([[Regularization]]): L1 is not differentiable at 0 and pushes weights **exactly** to zero (feature elimination); L2 only shrinks them.

## Distance

$$d(\boldsymbol{x},\boldsymbol{w})=\lVert \boldsymbol{x}-\boldsymbol{w}\rVert=\sqrt{\sum_i(x_i-w_i)^2},
\qquad
\lVert \boldsymbol{x}-\boldsymbol{w}\rVert^2=\sum_i(x_i-w_i)^2$$

Used as the similarity criterion by [[k-NN]], [[K-Means]], [[SOM]] and [[Vector Quantization]]. In all of them the winner/nearest search is done on the **squared** distance, because the root cannot change which one is smallest.

### The expansion

The identity to know:

$$\lVert \boldsymbol{a}\pm\boldsymbol{b}\rVert^2=\lVert \boldsymbol{a}\rVert^2\pm2\,\boldsymbol{a}^T\boldsymbol{b}+\lVert \boldsymbol{b}\rVert^2$$

It is the vector version of $(a\pm b)^2=a^2\pm2ab+b^2$, and it is doing the work in four different places:

- **[[Perceptron]] convergence:** expanding $\lVert \boldsymbol{w}(q)\rVert^2$ one update at a time gives the upper bound on the number of mistakes.
- **[[Kernel Methods]]:** distance in feature space with no feature space,
  $$\lVert \phi(\boldsymbol{x})-\phi(\boldsymbol{y})\rVert^2=k(\boldsymbol{x},\boldsymbol{x})-2k(\boldsymbol{x},\boldsymbol{y})+k(\boldsymbol{y},\boldsymbol{y}).$$
- **Winner search in [[K-Means]] / [[SOM]]:** $\lVert \boldsymbol{x}\rVert^2$ is the same for every unit, so it drops out and
  $$\arg\min_i\lVert \boldsymbol{x}-\boldsymbol{w}_i\rVert^2=\arg\max_i\Big(\boldsymbol{w}_i^T\boldsymbol{x}-\tfrac12\lVert \boldsymbol{w}_i\rVert^2\Big),$$
  turning a distance computation into a dot product.
- **[[Bias-Variance Tradeoff]]:** the same "expand the square" move, on scalars.

## Matrix products

Dimensions must chain: $(m\times k)(k\times n)=(m\times n)$. Checking them is the cheapest way to catch an error.

$\boldsymbol{A}\boldsymbol{x}$ has two readings, and the second is the useful one:

1. component $t$ is the dot product of row $t$ of $\boldsymbol{A}$ with $\boldsymbol{x}$;
2. $\boldsymbol{A}\boldsymbol{x}$ is a **linear combination of the columns** of $\boldsymbol{A}$, with the $x_i$ as coefficients.

Rules:

- **Not commutative:** $\boldsymbol{A}\boldsymbol{B}\ne\boldsymbol{B}\boldsymbol{A}$ in general (often not even the same shape).
- Associative and distributive.
- Transpose **reverses the order**: $(\boldsymbol{A}\boldsymbol{B})^T=\boldsymbol{B}^T\boldsymbol{A}^T$, and $(\boldsymbol{A}^T)^T=\boldsymbol{A}$.
- Inverse also reverses: $(\boldsymbol{A}\boldsymbol{B})^{-1}=\boldsymbol{B}^{-1}\boldsymbol{A}^{-1}$; it exists only for a square non-singular matrix.
- **Pseudoinverse** $\boldsymbol{A}^{+}$ generalizes the inverse to rectangular or singular matrices, computed via SVD $\to$ [[Normal Equation]], [[Randomized Neural Networks]].

### Outer product

$$\boldsymbol{a}\boldsymbol{b}^T \quad (m\times n \text{ matrix, rank }1)
\qquad\text{versus}\qquad
\boldsymbol{a}^T\boldsymbol{b}\quad(\text{scalar})$$

Same two vectors, the transpose on the other one, completely different object. In matrix form a weight update is an outer product, $\Delta \boldsymbol{W}=\eta\,\boldsymbol{\delta}\,\boldsymbol{o}^T$ - see [[Backpropagation]].

### The whole data set at once

- Predictions of a linear model: $\boldsymbol{X}\boldsymbol{w}$, i.e. $(l\times n)(n\times1)=l\times1$, one output per pattern.
- Residuals $\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}$, and the sum of squared errors is $\lVert \boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}\rVert^2$.
- $\boldsymbol{X}^T\boldsymbol{X}$ is $n\times n$ - **feature by feature** $\to$ [[Normal Equation]], [[Ridge Regression]].
- $\boldsymbol{X}\boldsymbol{X}^T$ is $l\times l$ - **pattern by pattern**: this is the Gram matrix, and with $\phi$ in place of the raw inputs it is the kernel matrix $\to$ [[Kernel Methods]].

> [!important]
> Which of the two you build decides the cost: $n$ features against $l$ patterns. Kernel methods deliberately work with the $l\times l$ one, which is why the dimension of the feature space stops mattering.

## The gradients you need

$\nabla_{\boldsymbol{w}}$ is the column vector of the $\partial/\partial w_i$.

| expression | $\nabla_{\boldsymbol{w}}$ |
|---|---|
| $\boldsymbol{a}^T\boldsymbol{w}$ | $\boldsymbol{a}$ |
| $\lVert \boldsymbol{w}\rVert^2=\boldsymbol{w}^T\boldsymbol{w}$ | $2\boldsymbol{w}$ |
| $\boldsymbol{w}^T\boldsymbol{A}\boldsymbol{w}$, $\boldsymbol{A}$ symmetric | $2\boldsymbol{A}\boldsymbol{w}$ |
| $\lVert \boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}\rVert^2$ | $-2\boldsymbol{X}^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w})$ |

Setting the last one to zero gives $\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{w}=\boldsymbol{X}^T\boldsymbol{y}$, the [[Normal Equation]]. Adding a penalty $\lambda\lVert \boldsymbol{w}\rVert^2$ adds $2\lambda\boldsymbol{w}$ to the gradient and gives

$$(\boldsymbol{X}^T\boldsymbol{X}+\lambda\boldsymbol{I})\,\boldsymbol{w}=\boldsymbol{X}^T\boldsymbol{y},$$

where the $\lambda\boldsymbol{I}$ is literally what makes the matrix invertible - see [[Ridge Regression]] and [[Regularization]].

## Geometry of a hyperplane

For $\boldsymbol{w}^T\boldsymbol{x}+w_0=0$:

- $\boldsymbol{w}$ is **orthogonal** to the hyperplane - it is the normal direction.
- Signed distance of a point from it: $\dfrac{\boldsymbol{w}^T\boldsymbol{x}+w_0}{\lVert \boldsymbol{w}\rVert}$; distance of the origin: $\dfrac{\lvert w_0\rvert}{\lVert \boldsymbol{w}\rVert}$.
- Scaling $(\boldsymbol{w},w_0)$ by any $c>0$ leaves the hyperplane and every decision unchanged - only the sign matters.

> [!note]
> That scaling freedom is exactly what [[SVM]] spends: fix the scale so that $\lvert \boldsymbol{w}^T\boldsymbol{x}+b\rvert=1$ at the closest points, and the margin becomes $\rho=2/\lVert \boldsymbol{w}\rVert$. "Maximize the margin" then *is* "minimize $\lVert \boldsymbol{w}\rVert^2$".

## Symmetric matrices, eigenvalues, definiteness

- **Symmetric:** $\boldsymbol{A}=\boldsymbol{A}^T$. Covariance matrices, Gram matrices and Hessians all are.
- **Eigenvalue / eigenvector:** $\boldsymbol{A}\boldsymbol{v}=\lambda\boldsymbol{v}$ - the directions the matrix only stretches.
- **Positive semi-definite (PSD):** $\boldsymbol{z}^T\boldsymbol{A}\boldsymbol{z}\ge0$ for every $\boldsymbol{z}$, equivalently all eigenvalues $\ge0$. A function is a **valid kernel** iff its matrix is symmetric PSD on every finite sample - see Mercer in [[Kernel Methods]].
- **Spectral radius:** $\rho(\boldsymbol{A})=\max_i\lvert\lambda_i\rvert$ - the contractivity condition of [[Echo State Networks]] and [[Graph Echo State Networks]] is stated on it.
- **Hessian:** the matrix of second derivatives, symmetric; its eigenvalues are the curvature of the error surface, which is why an ill-conditioned one gives the canyon that [[Gradient Descent]] crawls along and [[Momentum]] fixes.

## Traps

- $\boldsymbol{a}^T\boldsymbol{b}$ is a **scalar**, $\boldsymbol{a}\boldsymbol{b}^T$ is a **matrix**. Read the transpose, not the letters.
- $\boldsymbol{X}^T\boldsymbol{X}$ ($n\times n$) is not $\boldsymbol{X}\boldsymbol{X}^T$ ($l\times l$).
- $(\boldsymbol{A}\boldsymbol{B})^T=\boldsymbol{B}^T\boldsymbol{A}^T$ - the order flips, always.
- A squared-error loss is a **squared** norm. If a bias term in a derivation comes out unsquared, a term was dropped.
- Squaring never changes an $\arg\min$ over distances - but it does change a **mean**: $\mathbb{E}[Z^2]\ne\mathbb{E}[Z]^2$, and the gap is precisely the variance. That is the lemma behind [[Bias-Variance Tradeoff]].
- $w_{ij}$ is the weight **of unit $i$, from $j$** - not the other way round. See [[Notation]].

## Related

- [[Notation]] - which symbols are bold, and the per-model conventions.
- [[Linear Models]], [[Normal Equation]], [[Ridge Regression]] - where most of this is first used.
- [[Kernel Methods]], [[SVM]] - where the dot product becomes the whole point.
- [[Gradient Descent]], [[Backpropagation]] - where the gradients are used.
