---
tags: [machine-learning, models]
---

# SVM

Support Vector Machines. Three ideas stacked on top of each other:

1. a **linear machine** (an LTU, like the [[Perceptron]]) -
2. that maximizes the **separation margin** -
3. which turns out to be an automatic implementation of [[Structural Risk Minimization]].

Notation: $N$ examples (was $l$), $m$ input dimension (was $n$), $b$ instead of $w_0$, indices $i,j$ over patterns. See [[Notation]].

## The result

The trained model, in the form it is used:

$$
h(\boldsymbol{x})=\operatorname{sign}\big(g(\boldsymbol{x})\big),
\qquad
g(\boldsymbol{x})=\sum_{i=1}^{N_s}\alpha_{o,i}\,d_i\,k(\boldsymbol{x}_i,\boldsymbol{x})+b_o
$$

summed over the $N_s$ **support vectors** only, with $k(\boldsymbol{x}_i,\boldsymbol{x})=\boldsymbol{x}_i^T\boldsymbol{x}$ in the linear case. The margin it achieves is

$$
\boxed{\ \rho=\frac{2}{\lVert \boldsymbol{w}_o\rVert}\ }
\qquad\Longrightarrow\qquad
\text{maximize } \rho \iff \text{minimize } \lVert \boldsymbol{w}\rVert^2 .
$$

The two problems - what you write down, and what you actually solve:

| | primal (what it means) | dual (what you solve) |
|---|---|---|
| **objective** | minimize $\tfrac12\boldsymbol{w}^T\boldsymbol{w}+C\sum_i\xi_i$ | maximize $Q(\alpha)=\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_j d_i d_j\,\boldsymbol{x}_i^T\boldsymbol{x}_j$ |
| **constraints** | $d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\ge1-\xi_i$, $\ \xi_i\ge0$ | $\sum_i\alpha_i d_i=0$, $\ 0\le\alpha_i\le C$ |
| **unknowns** | $\boldsymbol{w}\in\mathbb{R}^m$, $b$, $\boldsymbol{\xi}$ | $\boldsymbol{\alpha}\in\mathbb{R}^N$ |
| **cost scales with** | the input dimension $m$ | the number of patterns $N$ |

**Hard margin** is the same thing with $\xi_i\equiv0$ and no upper bound on the multipliers (formally $C\to\infty$).

> [!important] The three facts to carry away
> 1. **Convex QP.** The objective is quadratic and convex, the constraints linear $\to$ training always reaches the **global** optimum. No local minima, no initialization, no learning rate.
> 2. **The data enter only through dot products** $\boldsymbol{x}_i^T\boldsymbol{x}_j$ $\to$ this is the door to [[Kernel Methods]].
> 3. **The solution is sparse.** $\alpha_i=0$ for every non-support vector, so the hyperplane depends only on the points at the boundary.

### Recipe

1. Choose $C$ and the kernel $k$ with its parameters.
2. Compute the kernel matrix $\boldsymbol{K}$, with $K_{ij}=k(\boldsymbol{x}_i,\boldsymbol{x}_j)$.
3. Solve the dual QP $\to$ $\{\alpha_{o,i}\}$. Generic QP solvers work; the dedicated one is **SMO** (Sequential Minimal Optimization).
4. $\boldsymbol{w}_o=\sum_i\alpha_{o,i}d_i\boldsymbol{x}_i$ - conceptually, but never computed in the kernel case.
5. $b_o=d_j-\sum_i\alpha_{o,i}d_i\,k(\boldsymbol{x}_i,\boldsymbol{x}_j)$ for any $j$ with $0<\alpha_j<C$, or averaged over all such $j$ for numerical stability.
6. Classify a new $\boldsymbol{x}$ with $\operatorname{sign}(g(\boldsymbol{x}))$.

> [!note]
> $\boldsymbol{w}_o$ is never needed explicitly - only the multipliers and the dot products. That is exactly what makes the kernel version possible: $\boldsymbol{w}$ lives in the feature space and can be enormous or infinite-dimensional, while $\boldsymbol{\alpha}$ always has $N$ components.

---

## Definitions

### Separating hyperplane

Given $T=\{(\boldsymbol{x}_i,d_i)\}_{i=1}^{N}$, find a hyperplane $\boldsymbol{w}^T\boldsymbol{x}+b=0$ with

$$
\boldsymbol{w}^T\boldsymbol{x}_i+b\ge0 \ \text{ for } d_i=+1, \qquad \boldsymbol{w}^T\boldsymbol{x}_i+b<0 \ \text{ for } d_i=-1 .
$$

- $g(\boldsymbol{x})=\boldsymbol{w}^T\boldsymbol{x}+b$ is the **discriminant function**;
- $h(\boldsymbol{x})=\operatorname{sign}(g(\boldsymbol{x}))$ is the hypothesis.

Same object as the LTU of [[Linear Models]] - what changes is *which* of the infinitely many separating hyperplanes we pick.

### Margin, and the optimal hyperplane

The **separation margin** $\rho$ is (twice) the distance between the hyperplane and the closest data point - a "safe zone". Different separating hyperplanes give different margins.

The **optimal hyperplane** $\boldsymbol{w}_o^T\boldsymbol{x}+b_o=0$ is the one that **maximizes $\rho$**.

### Canonical representation

Using the scaling freedom of $\boldsymbol{w}$ and $b$ (see [[Vector and Matrix Algebra]]), rescale so that the closest points satisfy $\lvert g(\boldsymbol{x}_i)\rvert=1$:

$$
\boldsymbol{w}^T\boldsymbol{x}_i+b\ge 1 \ \text{ if } d_i=+1,\qquad \boldsymbol{w}^T\boldsymbol{x}_i+b\le -1 \ \text{ if } d_i=-1
$$

which compacts to

$$
d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\ge 1 \qquad \forall i=1,\dots,N .
$$

> [!important] Support vector
> A **support vector** $\boldsymbol{x}^{(s)}$ is a point that satisfies this **exactly**: $d^{(s)}(\boldsymbol{w}^T\boldsymbol{x}^{(s)}+b)=1$. They are the closest data points to the hyperplane.

### Slack variables, and what they mean

Real data are noisy or not separable. Introduce non-negative **slack variables** $\xi_i\ge0$:

$$
d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\ge1-\xi_i \qquad \forall i .
$$

A support vector now satisfies it exactly: $d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)=1-\xi_i$.

| $\xi_i$ | where the point is |
|---|---|
| $\xi_i=0$ | outside or on the margin - correctly classified |
| $0<\xi_i\le1$ | inside the margin, still on the right side |
| $\xi_i>1$ | on the wrong side - a classification error |

$C$ is the **regularization hyperparameter** trading empirical risk against the capacity term:

- **Low $C$** $\to$ many training errors allowed $\to$ possible [[Underfitting]];
- **High $C$** $\to$ no training errors allowed $\to$ smaller margin $\to$ possible [[Overfitting]]. ($C\to\infty$ recovers the hard margin.)

> [!note]
> Admitting points inside the margin also lets us have a **larger** margin. The point is not only tolerating errors: noise tolerance *and* a bigger safe zone.

## Why it generalizes: SRM without hyperparameters

In the hard-margin case we **fixed the training error** to zero. Then minimizing $\lVert \boldsymbol{w}\rVert$ minimizes the VC dimension, hence the VC-confidence $\epsilon$ in

$$
R[h]\le R_{emp}[h]+\epsilon(VC,N,\delta) .
$$

> [!important] Vapnik's theorem
> Let $D$ be the diameter of the smallest ball around the data points. For the class of separating hyperplanes $\boldsymbol{w}^T\boldsymbol{x}+b=0$:
> $$VC \le \min\left(\left\lceil \frac{D^2}{\rho^2}\right\rceil,\ m_0\right)+1 .$$
> Since $D^2/\rho^2=\text{radius}^2\lVert \boldsymbol{w}\rVert^2$, constraining the search to the maximum-margin ("regularized") hyperplane can give a VC **smaller** than the $m_0+1$ of general hyperplanes.

So in the hard-margin case SVM performs an **automatic SRM**, folded into the training process itself, **with no hyperparameters**. That is the answer to "approach 2" of [[Structural Risk Minimization]]: fix the training error and let the algorithm minimize the capacity term.

> [!warning]
> With the soft margin **we lose the fully automatic SRM**, and Vapnik's theorem no longer holds (hard margin does not admit points inside the margin). $C$ must be chosen by [[Model Selection and Assessment|model selection]].

### Why it is an elegant approach

For linearly separable data there are many solutions; the maximum-margin one gives:

- a **unique** solution with zero errors for the binary classifier - not the case for the iterative [[Perceptron]] learning algorithm, nor for LMS;
- an automated SRM, without hyperparameters in the separable case;
- a **convex QP** solver instead of gradient descent, with a nice dual showing support vectors and dot products;
- a solution focused on "selected" training data:
  - *nice*: it does not depend on points far from the decision boundary,
  - *less nice*: it hopes the points **at** the boundary are not the noisiest ones.

## Non-linear: mapping to a feature space

Map the data into a high-dimensional **feature space** where they can be linearly separated:

$$
\Phi:\mathbb{R}^{m_0}\to\mathbb{R}^{m_1},\qquad \boldsymbol{x}\mapsto\Phi(\boldsymbol{x})
$$

Everything above goes through with $\Phi(\boldsymbol{x}_i)$ in place of $\boldsymbol{x}_i$; the hyperplane is $\boldsymbol{w}^T\Phi(\boldsymbol{x})+b=0$, and absorbing the bias with $\phi_0(\boldsymbol{x})=1$, $w_0=b$, the decision surface $\boldsymbol{w}^T\Phi(\boldsymbol{x})=0$ is exactly a **[[Linear Basis Expansion]]**.

**Example.** $\Phi:\mathbb{R}^2\to\mathbb{R}^3$, $(x_1,x_2)^T\mapsto(x_1^2,\sqrt2 x_1x_2,x_2^2)^T$: an ellipse in the input space becomes a plane in the second-order-monomial space.

Comments:

- Adding even one dimension can turn a non-linear problem into a linearly separable one.
- But finding the "best" transformation is **not automatic**, unless prior knowledge tells you the right feature space. General-purpose transformations often work well.
- **The choice of $\boldsymbol{\Phi}$ is the critical issue of the SVM.**

The theoretical backing is [[Cover's Theorem]]: patterns cast nonlinearly into a high-dimensional space are linearly separable with high probability.

The catch: a large basis expansion is computationally heavy and would normally overfit - *except* that here the complexity is ruled by the **margin**, not by the dimension of the space. What remains is the computational problem, and that is solved by [[Kernel Methods]].

## Architecture view

$$
h(\boldsymbol{x})=\operatorname{sign}\Big(\sum_{i=1}^{N}\alpha_i d_i\,k(\boldsymbol{x},\boldsymbol{x}_i)+b\Big)
$$

can be drawn as a network: an input layer of size $m_0$, a hidden layer of **inner-product kernels** $k(\boldsymbol{x},\boldsymbol{x}_i)$, and a linear output unit.

- **number of hidden units = number of support vectors**;
- but not the same kind of unit: hidden units in a [[Neural Networks|NN]] have **adaptive** free parameters, these do not. NN = *adaptive* basis functions, SVM = *fixed* (kernel-defined) ones.

---

## Derivations

### 1. The margin is $2/\lVert \boldsymbol{w}_o\rVert$

$\boldsymbol{w}_o$ is orthogonal to the hyperplane. Write a point as its projection plus a displacement along the normal,

$$
\boldsymbol{x}=\boldsymbol{x}_p+r\frac{\boldsymbol{w}_o}{\lVert \boldsymbol{w}_o\rVert},
$$

with $\boldsymbol{x}_p$ on the hyperplane (so $g(\boldsymbol{x}_p)=0$) and $r$ the signed distance. Evaluating $g$:

$$
g(\boldsymbol{x})=g(\boldsymbol{x}_p)+r\,\boldsymbol{w}_o^T\frac{\boldsymbol{w}_o}{\lVert \boldsymbol{w}_o\rVert}=r\frac{\lVert \boldsymbol{w}_o\rVert^2}{\lVert \boldsymbol{w}_o\rVert}=r\lVert \boldsymbol{w}_o\rVert
\quad\Longrightarrow\quad
r=\frac{g(\boldsymbol{x})}{\lVert \boldsymbol{w}_o\rVert}
$$

For a positive support vector the canonical form gives $g(\boldsymbol{x}^{(s)})=1$, so its distance is $1/\lVert \boldsymbol{w}_o\rVert$. The margin is twice that:

$$
\rho=\frac{2}{\lVert \boldsymbol{w}_o\rVert}.
$$

Maximizing $\rho$ is therefore minimizing $\lVert \boldsymbol{w}\rVert$, and we minimize $\tfrac12\lVert \boldsymbol{w}\rVert^2$ instead - same argmin, differentiable, and the $\tfrac12$ cancels the 2 from the derivative.

### 2. Primal $\to$ Lagrangian $\to$ dual

Primal (hard margin): minimize $\Psi(\boldsymbol{w})=\tfrac12\boldsymbol{w}^T\boldsymbol{w}$ subject to $d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\ge1$ for all $i$.

Build the Lagrangian, one multiplier $\alpha_i\ge0$ per constraint:

$$
J(\boldsymbol{w},b,\boldsymbol{\alpha})=\tfrac12\boldsymbol{w}^T\boldsymbol{w}-\sum_{i=1}^{N}\alpha_i\big(d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)-1\big)
$$

to be **minimized** w.r.t. $\boldsymbol{w},b$ and **maximized** w.r.t. $\boldsymbol{\alpha}$ - the solution is a **saddle point** of $J$. Setting the derivatives to zero:

$$
\frac{\partial J}{\partial \boldsymbol{w}}=0 \Rightarrow \boldsymbol{w}=\sum_{i=1}^{N}\alpha_i d_i \boldsymbol{x}_i,
\qquad
\frac{\partial J}{\partial b}=0 \Rightarrow \sum_{i=1}^{N}\alpha_i d_i=0 .
$$

The first is the **representer** result: the optimal $\boldsymbol{w}$ is a linear combination of the training points. Substituting both back into $J$ eliminates $\boldsymbol{w}$ and $b$ and leaves only the multipliers:

> **Dual problem.** Maximize
> $$Q(\boldsymbol{\alpha})=\sum_{i=1}^{N}\alpha_i-\frac12\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j d_i d_j\ \underbrace{\boldsymbol{x}_i^T\boldsymbol{x}_j}_{\textbf{dot product}}$$
> subject to $\alpha_i\ge0$ and $\sum_i\alpha_i d_i=0$.

Two things to notice, and they are the whole story of what comes later:

- the data enter **only through dot products** $\boldsymbol{x}_i^T\boldsymbol{x}_j$ $\to$ the door for [[Kernel Methods]];
- solving the dual scales with the **number of patterns $N$**, not with the space dimension.

### 3. Kuhn-Tucker: why the solution is sparse

At the optimum the complementarity condition holds for every constraint:

$$
\alpha_i\big(d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)-1\big)=0 \quad\forall i
$$

A product is zero only if one factor is. Reading both directions:

- if $\alpha_i>0$ then $d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)=1$, i.e. $\boldsymbol{x}_i$ **is** a support vector;
- if $\boldsymbol{x}_i$ is not a support vector, its constraint is strictly satisfied, so $\alpha_i=0$.

> [!important]
> The solution is **sparse in $\boldsymbol{\alpha}$**, and the hyperplane depends **only on the support vectors**:
> $$\boldsymbol{w}_o=\sum_{i=1}^{N_s}\alpha_{o,i}d_i\boldsymbol{x}_i .$$

### 4. Soft margin: where the slacks go

Primal: minimize $\Psi(\boldsymbol{w},\boldsymbol{\xi})=\tfrac12\boldsymbol{w}^T\boldsymbol{w}+C\sum_{i=1}^{N}\xi_i$ subject to $d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\ge1-\xi_i$ and $\xi_i\ge0$.

Now there are **two** families of multipliers: $\alpha_i$ for the margin constraints and $\mu_i$ for $\xi_i\ge0$. Differentiating the Lagrangian w.r.t. $\xi_i$ gives

$$
C-\alpha_i-\mu_i=0 \quad\Longrightarrow\quad \alpha_i=C-\mu_i,
$$

and since $\mu_i\ge0$ this is exactly $\alpha_i\le C$. So the slacks **cancel out of the dual** and leave a single trace: an upper bound on the multipliers.

$$
Q(\boldsymbol{\alpha})=\sum_i \alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_j d_i d_j\,\boldsymbol{x}_i^T\boldsymbol{x}_j,
\qquad
\sum_i\alpha_i d_i=0,
\qquad
\boxed{0\le\alpha_i\le C}
$$

The objective is **identical** to the hard-margin one - only the box constraint changes.

The KT conditions become $\alpha_i(d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)+\xi_i-1)=0$ and $\mu_i\xi_i=0$, which classify every point:

| $\alpha_i$ | $\mu_i=C-\alpha_i$ | $\xi_i$ | position |
|---|---|---|---|
| $\alpha_i=0$ | $>0$ | $0$ | outside the margin, not a support vector |
| $0<\alpha_i<C$ | $>0$ | $0$ | **on the edge** of the margin |
| $\alpha_i=C$ | $0$ | $\ge0$ | **inside** the margin, or misclassified |

### 5. Recovering $b_o$

$b$ vanished from the dual, so it has to be recovered afterwards. Take any $j$ with $0<\alpha_j<C$: by the table above such a point has $\xi_j=0$ and sits exactly on the margin edge, so $d_j(\boldsymbol{w}_o^T\boldsymbol{x}_j+b_o)=1$, and multiplying by $d_j$ (with $d_j^2=1$):

$$
b_o=d_j-\boldsymbol{w}_o^T\boldsymbol{x}_j=d_j-\sum_i \alpha_{o,i}d_i\,\boldsymbol{x}_i^T\boldsymbol{x}_j .
$$

In the hard-margin case every support vector qualifies, and for a positive one this is $b_o=1-\boldsymbol{w}_o^T\boldsymbol{x}^{(s)}$. In practice average over all valid $j$ for numerical stability.

> [!warning]
> Only points with $0<\alpha_j<C$ can be used. A point with $\alpha_j=C$ may have $\xi_j>0$, so its constraint is not tight and it gives the wrong $b$.

---

## Summary of the characteristics

**Pros**

- Regularization is embedded in the optimization problem (the margin).
- Approximation of theoretical SRM (bounded VC-dim of the hyperplane, minimized by the solution - hard margin, no kernel).
- **Convex** problem: training always finds a global minimum.
- Implicit feature transformation via kernels: a rich set of non-linear decision functions in the input space, while still doing a *linear* separation in another space.
- Linear model at heart: simple, compact.

**Cons**

- You must choose the kernel and its parameters (and $C$, and $\epsilon$ for regression).
- **Batch** algorithm.
- Very large problems were computationally intractable (>20,000 examples was very hard to solve exactly with standard approaches); many solutions exist nowadays, including gradient-descent approaches.

## Critical view

> There is no theory guaranteeing that a given family of SVMs will have high accuracy on a given problem.

The nice properties of the hard-margin classifier **do not extend** to soft margin and kernels: $C$ and the kernel can lead to **infinite VC-dim**.

- A gaussian RBF kernel with sufficiently small width $\sigma$ can classify an arbitrarily large number of training points correctly (like 1-NN: only the closest support vector contributes) $\to$ infinite VC-dim unless regularized.
- With large width, all the support vectors are considered and you get a sort of global average $\to$ low VC-dim.
- So **controlling the width controls the VC-dim** - and small $\sigma$ (narrow peaked kernels) requires a **lower $C$** (more regularization). The best $C$ depends on the kernel hyperparameters: they must be searched **together**, see [[Grid Search]] and [[Validation Pitfalls]].

Other honest points:

- **Number of support vectors:** assumed low in the discussion, not always true. In the classic mixture example, 42% of the data are support vectors with $C$ tuned to the best performance, and 45% with the RBF kernel.
- **Hyperparameter fitting can overfit**: use a validation set for $C$/kernel and then a separate test set, or double/nested CV. See [[Validation Schemas]].
- **Performance:** often good, but the comparison is not always favourable w.r.t. other ML methods - on image and speech tasks in particular, deep models do better. See [[Deep Learning]].

### Typical misconceptions

1. *"SVM can be used with default hyperparameter values."* No - see above.
2. *"I choose SVM because it does not fall into overfitting."* No - soft margin and kernels can.
3. *"SVM has no problem with high input dimension - does it solve the curse of dimensionality?"* **No.** It deals well with high dimensionality in the **feature** space, not in the **input** space. See [[Curse of Dimensionality]].
4. *"With SVM you automatically find the number of units."* Only in the picture-view sense above; it is the QP solution that decides how many support vectors survive, not a capacity search.

## Related

- [[Kernel Methods]] - the kernel trick and its life beyond SVM.
- [[SVM Regression]] - the $\epsilon$-insensitive version.
- [[Structural Risk Minimization]], [[VC Dimension]], [[Cover's Theorem]]
- [[Linear Models]], [[Linear Basis Expansion]], [[Perceptron]], [[Neural Networks]]
- [[Vector and Matrix Algebra]] - the scaling freedom, the margin geometry, the dot products.
