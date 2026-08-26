---
tags: [machine-learning, theory]
---

# Structural Risk Minimization

SRM uses the [[VC Dimension]] as the knob to minimize the generalization bound

$$
R \le R_{emp}+\epsilon(1/l,VC,1/\delta).
$$

Assuming finite VC, define a **nested structure** $H_1\subset H_2\subset\dots$ ordered by VC-dim, and pick the element with the best bound - not the lowest training error.

- Growing VC $\to$ $R_{emp}$ decreases, VC-confidence increases.
- The inductive principle of SRM: the optimal complexity is the minimum of the bound.

## Examples of nested structures

- [[Neural Networks]] with an increasing number of hidden units (roughly), and also the number of epochs - a NN is a **variable-size $H$**.
- Polynomials of increasing degree $M$.
- Increasing $c$ in $\lVert w\rVert<c$, i.e. decreasing $\lambda$ in [[Regularization]].
- Increasing number of nodes in a decision tree (roughly).

> [!important]
> The important hyperparameters are exactly the ones related to the VC-dim.

## Two practical approaches

1. **Fix the structure/complexity** (and hence the VC-confidence), then minimize the training error. This is what we do with NN. Note that the training heuristics can already implement an implicit SRM ([[Early Stopping]]), or the explicit Tikhonov version where the loss contains both $R_{emp}$ and the complexity term.
2. **Fix the training error** and automatically minimize the VC-confidence $\to$ this is [[SVM]]: with a separable problem the training error is zero by construction, and minimizing $\lVert w\rVert$ (i.e. maximizing the margin) minimizes the VC-dim bound $VC\le\min(\lceil D^2/\rho^2\rceil,m_0)+1$. SRM folded into the training process, with no hyperparameters - until the soft margin brings $C$ back.

> [!warning]
> The VC-confidence can be very conservative - hundreds of times larger than the empirical overfitting effect. Useful for ranking models, not for a reliable value of the generalization error.

Related: [[Statistical Learning Theory]].
