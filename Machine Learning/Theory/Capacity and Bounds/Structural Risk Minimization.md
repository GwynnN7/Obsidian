---
tags: [machine-learning, theory]
---

# Structural Risk Minimization

SRM uses the [[VC Dimension]] as the knob to minimize the generalization bound

$$
R \le R_{emp}+\epsilon(1/l,VC,1/\delta).
$$

## The structure

Assuming finite VC, define a **nested structure** of hypothesis spaces ordered by capacity:

$$
H_1\subseteq H_2\subseteq\dots\subseteq H_n,
\qquad
VC(H_1)\le VC(H_2)\le\dots\le VC(H_n),
$$

and pick the element with the best bound - not the one with the lowest training error.

- Growing VC $\to$ $R_{emp}$ decreases, VC-confidence increases.
- The inductive principle of SRM: the optimal complexity is the **minimum of the bound**.

Note the two conditions are separate requirements: the spaces must be nested *and* ordered by VC-dim. The nesting is what makes "increase the complexity by one step" a well-defined move.

## Examples of nested structures

- [[Neural Networks]] with an increasing number of hidden units (roughly), and also the number of epochs - the size of the weights matters too, so a NN is a **variable-size $H$**.
- Polynomials of increasing degree $M$.
- Increasing $c$ in $\lVert \boldsymbol{w}\rVert<c$, i.e. decreasing $\lambda$ in [[Regularization]].
- Increasing number of nodes in a decision tree (roughly).

> [!important]
> The important hyperparameters are exactly the ones related to the VC-dim.

## Model selection with the bound

For each $H_i$ in the structure, tabulate three columns:

| $R_{emp}$     | measured after training on $H_i$ - **decreases** down the structure                                |
| ------------- | -------------------------------------------------------------------------------------------------- |
| VC-confidence | computed from $VC(H_i)$, $l$, $\delta$ - **increases** down the structure, and before any training |
| bound on $R$  | their sum                                                                                          |

Pick the row with the smallest bound. The two columns move in opposite directions, so the sum has a minimum somewhere in the middle - typically the first row where the VC-confidence starts to grow faster than the training error falls.

> [!warning]
> The VC-confidence can be very conservative - hundreds of times larger than the empirical overfitting effect. Useful for **ranking** models, not for a reliable value of the generalization error.

## Two practical approaches

1. **Fix the structure/complexity** (and hence the VC-confidence), then minimize the training error. This is what we do with NN. Note that the training heuristics can already implement an implicit SRM ([[Early Stopping]]), or the explicit Tikhonov version where the loss contains both $R_{emp}$ and the complexity term.
2. **Fix the training error** and automatically minimize the VC-confidence $\to$ this is [[SVM]]: with a separable problem the training error is zero by construction, and minimizing $\lVert \boldsymbol{w}\rVert$ (i.e. maximizing the margin) minimizes the VC-dim bound $VC\le\min(\lceil D^2/\rho^2\rceil,m_0)+1$. SRM folded into the training process, with no hyperparameters - until the soft margin brings $C$ back.

Related: [[Statistical Learning Theory]], [[Model Selection and Assessment]].
