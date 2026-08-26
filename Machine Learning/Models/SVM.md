---
tags: [machine-learning, models]
---

# SVM

Support Vector Machines. Three ideas stacked on top of each other:

1. a **linear machine** (an LTU, like the [[Perceptron]]) -
2. that maximizes the **separation margin** -
3. which turns out to be an automatic implementation of [[Structural Risk Minimization]].

Notation here follows: $N$ examples (was $l$), $m$ input dimension (was $n$), $b$ instead of $w_0$, indices $i,j$ over patterns. See [[Notation]].

We start with the **hard margin** case, assuming linearly separable data and no errors.

---

## Separating hyperplane

Given $T=\{(x_i,d_i)\}_{i=1}^{N}$, find a hyperplane $w^Tx+b=0$ with

$$
w^Tx_i+b\ge0 \ \text{ for } d_i=+1, \qquad w^Tx_i+b<0 \ \text{ for } d_i=-1 .
$$

- $g(x)=w^Tx+b$ is the **discriminant function**;
- $h(x)=\operatorname{sign}(g(x))$ is the hypothesis.

Same object as the LTU of [[Linear Models]] - what changes is *which* of the infinitely many separating hyperplanes we pick.

## Margin, and the optimal hyperplane

The **separation margin** $\rho$ is (twice) the distance between the hyperplane and the closest data point - a "safe zone". Different separating hyperplanes give different margins.

The **optimal hyperplane** $w_o^Tx+b_o=0$ is the one that **maximizes $\rho$**.

### Canonical representation

Using the scaling freedom of $w$ and $b$ (see [[Linear Models]]), rescale so that the closest points satisfy $\lvert g(x_i)\rvert=1$:

$$
w^Tx_i+b\ge 1 \ \text{ if } d_i=+1,\qquad w^Tx_i+b\le -1 \ \text{ if } d_i=-1
$$

which compacts to

$$
d_i(w^Tx_i+b)\ge 1 \qquad \forall i=1,\dots,N .
$$

> [!important] Support vector
> A **support vector** $x^{(s)}$ is a point that satisfies this **exactly**: $d^{(s)}(w^Tx^{(s)}+b)=1$. They are the closest data points to the hyperplane.

### Deriving $\rho=2/\lVert w_o\rVert$

$w_o$ is orthogonal to the hyperplane. Writing a point as $x=x_p+r\frac{w_o}{\lVert w_o\rVert}$ (with $x_p$ its projection and $r$ the signed distance) and evaluating $g$:

$$
g(x)=g(x_p)+r\,w_o^T\frac{w_o}{\lVert w_o\rVert}=r\frac{\lVert w_o\rVert^2}{\lVert w_o\rVert}=r\lVert w_o\rVert
\quad\Longrightarrow\quad
r=\frac{g(x)}{\lVert w_o\rVert}
$$

For a positive support vector $g(x^{(s)})=1$, so its distance is $\frac{1}{\lVert w_o\rVert}=\frac{\rho}{2}$, giving

$$
\boxed{\ \rho=\frac{2}{\lVert w_o\rVert}\ }
\qquad\Longrightarrow\qquad
\text{maximize } \rho \iff \text{minimize } \lVert w\rVert .
$$

## The optimization problem

### Primal form (hard margin)

Find $w,b$ minimizing

$$
\Psi(w)=\tfrac12 w^Tw
$$

subject to (zero classification errors)

$$
d_i(w^Tx_i+b)\ge1 \quad \forall i=1,\dots,N .
$$

- $\Psi$ is **quadratic and convex** in $w$, the constraints are **linear** $\to$ a convex quadratic program: training always finds a **global** minimum.
- The cost of solving it in this form scales with the **input dimension $m$**.

### Dual form

Build the Lagrangian with multipliers $\alpha_i\ge0$:

$$
J(w,b,\alpha)=\tfrac12 w^Tw-\sum_{i=1}^{N}\alpha_i\big(d_i(w^Tx_i+b)-1\big)
$$

to be minimized w.r.t. $w,b$ and maximized w.r.t. $\alpha$ - the solution is a **saddle point** of $J$. Setting the derivatives to zero:

$$
\frac{\partial J}{\partial w}=0 \Rightarrow w=\sum_{i=1}^{N}\alpha_i d_i x_i,
\qquad
\frac{\partial J}{\partial b}=0 \Rightarrow \sum_{i=1}^{N}\alpha_i d_i=0 .
$$

Substituting back gives the dual:

> **Dual problem.** Maximize
> $$Q(\alpha)=\sum_{i=1}^{N}\alpha_i-\frac12\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j d_i d_j\ \underbrace{x_i^Tx_j}_{\textbf{dot product}}$$
> subject to $\alpha_i\ge0$ and $\sum_i\alpha_id_i=0$.

Two things to notice, and they are the whole story of what comes later:

- the data enter **only through dot products** $x_i^Tx_j$ $\to$ this is the door for [[Kernel Methods]];
- solving the dual scales with the **number of patterns $N$**, not with the space dimension. (Solvable by generic QP, or by dedicated methods such as **SMO** - Sequential Minimal Optimization.)

### Kuhn-Tucker conditions and sparsity

$$
\alpha_i\big(d_i(w^Tx_i+b)-1\big)=0 \quad\forall i
$$

- if $\alpha_i>0$ then $d_i(w^Tx_i+b)=1$, i.e. $x_i$ **is** a support vector;
- if $x_i$ is not a support vector then $\alpha_i=0$.

> [!important]
> The solution is **sparse in $\alpha$**, and the hyperplane depends **only on the support vectors**:
> $$w_o=\sum_{i=1}^{N_s}\alpha_{o,i}d_ix_i .$$

## Using the model

1. Solve the dual $\to$ $\{\alpha_{o,i}\}$.
2. $w_o=\sum_i \alpha_{o,i}d_ix_i$.
3. $b_o=1-w_o^Tx^{(s)}$ for a positive support vector (or average over all of them, for numerical stability).
4. For a new $x$: compute $g(x)=\sum_i\alpha_{o,i}d_i\,x_i^Tx+b_o$ and classify with $\operatorname{sign}(g(x))$.

> [!note]
> We never need $w_o$ explicitly - only the multipliers and the dot products - and the sum runs over the $N_s$ support vectors. That is exactly what makes the kernel version possible.

---

## Why it generalizes: SRM without hyperparameters

We **fixed the training error** to zero (separable case). Then minimizing $\lVert w\rVert$ minimizes the VC dimension, hence the VC-confidence $\epsilon$ in

$$
R[h]\le R_{emp}[h]+\epsilon(VC,N,\delta) .
$$

> [!important] Vapnik's theorem
> Let $D$ be the diameter of the smallest ball around the data points. For the class of separating hyperplanes $w^Tx+b=0$:
> $$VC \le \min\left(\left\lceil \frac{D^2}{\rho^2}\right\rceil,\ m_0\right)+1 .$$
> Since $D^2/\rho^2=\text{radius}^2\lVert w\rVert^2$, constraining the search to the maximum-margin ("regularized") hyperplane can give a VC **smaller** than the $m_0+1$ of general hyperplanes.

So in the hard-margin case SVM performs an **automatic SRM**, folded into the training process itself, **with no hyperparameters**. That is the answer to "approach 2" of [[Structural Risk Minimization]]: fix the training error and let the algorithm minimize the capacity term.

### Why it is an elegant approach

For linearly separable data there are many solutions; the maximum-margin one gives:

- a **unique** solution with zero errors for the binary classifier - not the case for the iterative [[Perceptron]] learning algorithm, nor for LMS;
- an automated SRM, without hyperparameters in the separable case;
- a **convex QP** solver instead of gradient descent, with a nice dual showing support vectors and dot products;
- a solution focused on "selected" training data:
  - *nice*: it does not depend on points far from the decision boundary,
  - *less nice*: it hopes the points **at** the boundary are not the noisiest ones.

---

## Soft margin: admitting errors

Real data are noisy or not separable. Introduce non-negative **slack variables** $\xi_i\ge0$:

$$
d_i(w^Tx_i+b)\ge1-\xi_i \qquad \forall i .
$$

A support vector now satisfies it exactly: $d_i(w^Tx_i+b)=1-\xi_i$.

- $\xi_i=0$: outside or on the margin - correctly classified.
- $0<\xi_i\le1$: inside the margin, still on the right side.
- $\xi_i>1$: on the wrong side - a classification error.

> [!note]
> Admitting points inside the margin also lets us have a **larger** margin. The point is not only tolerating errors: noise tolerance *and* a bigger safe zone.

### Primal (soft margin)

$$
\Psi(w,\xi)=\tfrac12 w^Tw + C\sum_{i=1}^{N}\xi_i
$$

subject to $d_i(w^Tx_i+b)\ge1-\xi_i$ and $\xi_i\ge0$.

**$C$ is a regularization hyperparameter** trading off empirical risk against the capacity term:

- **Low $C$** $\to$ many training errors allowed $\to$ possible [[Underfitting]];
- **High $C$** $\to$ no training errors allowed $\to$ smaller margin $\to$ possible [[Overfitting]]. ($C\to\infty$ recovers the hard margin.)

> [!warning]
> With the soft margin **we lose the fully automatic SRM**, and Vapnik's theorem above no longer holds (hard margin does not admit points inside the margin). $C$ must be chosen by [[Model Selection and Assessment|model selection]].

### Dual (soft margin)

Identical objective, one changed constraint:

$$
Q(\alpha)=\sum_i \alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_j d_i d_j\,x_i^Tx_j,
\qquad
\sum_i\alpha_id_i=0,
\qquad
\boxed{0\le\alpha_i\le C}
$$

The slack variables disappear from the dual; the only trace they leave is the **upper bound $C$ on the multipliers**.

KT conditions become $\alpha_i(d_i(w^Tx_i+b)+\xi_i-1)=0$ and $\mu_i\xi_i=0$ (the $\mu_i$ enforce $\xi_i\ge0$), which read as:

- $0<\alpha_i<C \Rightarrow \xi_i=0$: the point sits **on the edge of the margin**;
- $\alpha_i=C \Rightarrow \xi_i\ge0$: the point is **inside the margin** (or misclassified).

Then $b_o=d_j-\sum_i \alpha_{o,i}d_i x_i^Tx_j$ for any $j$ with $0<\alpha_j<C$ (or averaged).

---

## Non-linear: mapping to a feature space

Map the data into a high-dimensional **feature space** where they can be linearly separated:

$$
\Phi:\mathbb{R}^{m_0}\to\mathbb{R}^{m_1},\qquad x\mapsto\Phi(x)
$$

Everything above goes through with $\Phi(x_i)$ in place of $x_i$; the hyperplane is $w^T\Phi(x)+b=0$, and absorbing the bias with $\phi_0(x)=1$, $w(0)=b$, the decision surface $w^T\Phi(x)=0$ is exactly a **[[Linear Basis Expansion]]**.

**Example.** $\Phi:\mathbb{R}^2\to\mathbb{R}^3$, $(x_1,x_2)^T\mapsto(x_1^2,\sqrt2 x_1x_2,x_2^2)^T$: an ellipse in the input space becomes a plane in the second-order-monomial space.

Comments:

- Adding even one dimension can turn a non-linear problem into a linearly separable one.
- But finding the "best" transformation is **not automatic**, unless prior knowledge tells you the right feature space. General-purpose transformations often work well.
- **The choice of $\Phi$ is the critical issue of the SVM.**

The theoretical backing is [[Cover's Theorem]]: patterns cast nonlinearly into a high-dimensional space are linearly separable with high probability.

The catch: a large basis expansion is computationally heavy and would normally overfit - *except* that here the complexity is ruled by the **margin**, not by the dimension of the space. What remains is the computational problem, and that is solved by [[Kernel Methods]].

## Architecture view

$$
h(x)=\operatorname{sign}\Big(\sum_{i=1}^{N}\alpha_i d_i\,k(x,x_i)+b\Big)
$$

can be drawn as a network: an input layer of size $m_0$, a hidden layer of **inner-product kernels** $k(x,x_i)$, and a linear output unit.

- **number of hidden units = number of support vectors**;
- but not the same kind of unit: hidden units in a [[Neural Networks|NN]] have **adaptive** free parameters, these do not. NN = *adaptive* basis functions, SVM = *fixed* (kernel-defined) ones.

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

> "Currently there exists no theory which guarantees that a given family of SVMs will have high accuracy on a given problem" (Burges, 1998).

The nice properties of the hard-margin classifier **do not extend** to soft margin and kernels: $C$ and the kernel can lead to **infinite VC-dim**.

- A gaussian RBF kernel with sufficiently small width $\sigma$ can classify an arbitrarily large number of training points correctly (like 1-NN: only the closest support vector contributes) $\to$ infinite VC-dim unless regularized.
- With large width, all the support vectors are considered and you get a sort of global average $\to$ low VC-dim.
- So **controlling the width controls the VC-dim** - and small $\sigma$ (narrow peaked kernels) requires a **lower $C$** (more regularization). The best $C$ depends on the kernel hyperparameters: they must be searched **together**, see [[Grid Search]] and [[Validation Pitfalls]].

Other honest points:

- **Number of support vectors:** assumed low in the discussion, not always true. In the classic mixture example, 42% of the data are support vectors with $C$ tuned to the best performance, and 45% with the RBF kernel.
- **Hyperparameter fitting can overfit**: use a validation set for $C$/kernel and then a separate test set, or double/nested CV. See [[Validation Schemas]].
- **Performance:** often good, but the comparison is not always favourable w.r.t. other ML methods. Since 2012/13, deep models have largely outperformed the previous records on image and speech recognition - see [[Deep Learning]].

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
