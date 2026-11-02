---
tags: [machine-learning, models]
---

# Kernel Methods

Compute a dot product in a high-dimensional feature space **without ever going there**.

## The result

$$
k(\boldsymbol{x},\boldsymbol{y})=\Phi^T(\boldsymbol{x})\,\Phi(\boldsymbol{y}),
\qquad
k:\mathbb{R}^{m_0}\times\mathbb{R}^{m_0}\to\mathbb{R}
$$

$k$ is the **inner-product kernel**: symmetric, $k(\boldsymbol{x},\boldsymbol{y})=k(\boldsymbol{y},\boldsymbol{x})$. Collecting all pairs gives the $N\times N$ **kernel (Gram) matrix**

$$
\boldsymbol{K}=\{k(\boldsymbol{x}_i,\boldsymbol{x}_j)\}_{i,j=1}^{N}.
$$

> [!important] Mercer's theorem - which functions are valid
> Not every function of two arguments is a kernel. $k$ corresponds to an inner product in *some* feature space **iff** its kernel matrix is symmetric **positive semi-definite** on every finite sample - equivalently, non-negative eigenvalues.

The standard kernels:

| Kernel | Form | Hyperparameter |
|---|---|---|
| **Polynomial learning machine** | $k(\boldsymbol{x},\boldsymbol{x}_i)=(\boldsymbol{x}^T\boldsymbol{x}_i+1)^p$ | degree $p$ |
| **RBF / gaussian** | $k(\boldsymbol{x},\boldsymbol{x}_i)=e^{-\frac{1}{2\sigma^2}\lVert \boldsymbol{x}-\boldsymbol{x}_i\rVert^2}$ | width $\sigma$ (often $\gamma=1/2\sigma^2$) |
| **Two-layer perceptron** | $k(\boldsymbol{x},\boldsymbol{x}_i)=\tanh(\beta_0\boldsymbol{x}^T\boldsymbol{x}_i+\beta_1)$ | $\beta_0>0$, $\beta_1<0$ |

And the distance the kernel induces, still without $\Phi$:

$$
d_k(\boldsymbol{s},\boldsymbol{t})=\lVert \Phi(\boldsymbol{s})-\Phi(\boldsymbol{t})\rVert^2=k(\boldsymbol{s},\boldsymbol{s})-2k(\boldsymbol{s},\boldsymbol{t})+k(\boldsymbol{t},\boldsymbol{t})
$$

(just the expansion of $\lVert \boldsymbol{a}-\boldsymbol{b}\rVert^2$, see [[Vector and Matrix Algebra]]).

> [!important]
> Everything the learner needs about the data is in $\boldsymbol{K}$. The feature mapping is never computed and the feature space is never named.

## The problem it solves

In the dual form of [[SVM]] the data appear **only** through dot products. After mapping into a feature space, the weight vector is a linear combination of the feature vectors:

$$
\boldsymbol{w}=\sum_{i=1}^{N}\alpha_i d_i\Phi(\boldsymbol{x}_i)
\qquad\Longrightarrow\qquad
\sum_{i=1}^{N}\alpha_i d_i\underbrace{\Phi^T(\boldsymbol{x}_i)\Phi(\boldsymbol{x})}_{\text{dot product}}=0
$$

The dot product is now in the feature space, and **evaluating $\Phi(\boldsymbol{x})$ could be intractable** - a large or even infinite basis expansion. The kernel makes that evaluation unnecessary.

## The trick, worked

With $\Phi((x_1,x_2)^T)=(x_1^2,\sqrt2x_1x_2,x_2^2)^T$:

$$
\Phi^T(\boldsymbol{x})\Phi(\boldsymbol{y})=x_1^2y_1^2+2x_1x_2y_1y_2+x_2^2y_2^2=(x_1y_1+x_2y_2)^2=(\boldsymbol{x}^T\boldsymbol{y})^2=k(\boldsymbol{x},\boldsymbol{y})
$$

> [!important]
> The dot product in the 3D feature space is evaluated with a 2D computation on the **input** patterns. Nothing was mapped; the middle expression is only there to show the two ends agree.

## Closure properties

If $k_1$ and $k_2$ are kernels over $\mathbb{R}^{m_0}\times\mathbb{R}^{m_0}$, so are:

- $k_1(\boldsymbol{x},\boldsymbol{y})+k_2(\boldsymbol{x},\boldsymbol{y})$
- $\alpha\,k_1(\boldsymbol{x},\boldsymbol{y})$ for $\alpha\in\mathbb{R}_+$
- $k_1(\boldsymbol{x},\boldsymbol{y})\,k_2(\boldsymbol{x},\boldsymbol{y})$

Handy for building new kernels out of known ones - and the cheapest way to prove a new one valid without touching Mercer directly.

## Notes on the standard kernels

- For polynomial machines and RBF you **always** get an inner-product kernel; for the two-layer perceptron, Mercer's theorem holds only for some choices of $\beta_0,\beta_1$.
- An RBF kernel always corresponds to a feature space with an **infinite** number of dimensions.
- Narrow peaked kernels (small $\sigma$) mean the reply for $\boldsymbol{x}_i$ is essentially only $d_i$ - which is [[k-NN]] with $k=1$. See the VC-dim discussion in [[SVM]].

## Kernels as similarity

Similar objects (small $d_k$) $\to$ high kernel value.

> The idea underpinning the use of a kernel function is to focus on the representation and processing of **pairwise comparisons** rather than on the representation of a single object. For some domains it is easier to compare two objects than to define an abstract feature space for one.

## Designing kernels

You need a **good** kernel for the specific domain, and the inputs need not be vectors at all:

- kernels for **strings, trees, graphs** (e.g. edit-distance kernels for bioinformatics, suffix-tree kernels for strings) $\to$ [[Structured Domain Learning]];
- ad hoc for the data domain and task (NLP, bioinformatics, ...);
- efficiency matters - the comparison can often be computed in clever ways;
- **adaptive kernels** are still a research issue.

> [!important]
> The background knowledge is **separated from the solver**: $\boldsymbol{K}$ can change without changing the learning machine. An SVM is largely characterized by the choice of its kernel - and the best choice for a given problem is still an open research question.
>
> Compare with [[Neural Networks]], where the basis functions themselves are **adapted to the data** by fitting the $\boldsymbol{w}$ inside $\phi$. Fixed-but-modular (kernels) vs learned (NN) - the same "random features vs learned features" axis as in [[Randomized Neural Networks]].

## Beyond SVM

**Kernel trick $\to$ kernel methods.** "Kernelization" of traditional approaches: wherever your model contains a dot product or a similarity measure, you can let it operate in an implicit high-dimensional space just by specifying a kernel.

- A very efficient fixed high-dimensional [[Linear Basis Expansion]].
- Modular choice of the basis functions, through the kernel.
- A unified theoretical framework for general dot-product functions.

Related: [[SVM]], [[Cover's Theorem]], [[SVM Regression]], [[Structured Domain Learning]], [[Vector and Matrix Algebra]].
