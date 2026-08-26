---
tags: [machine-learning, theory, models]
---

# Curse of Dimensionality

As the number of input dimensions grows, the volume of the problem space grows so fast that the available data become sparse.

Three manifestations:

1. **Hard to find nearby points in high $n$.** Neighborhoods become spatially large and stop being local; shrinking them means lowering $k$ and raising the variance.
   In a unit cube, to capture a fraction $r$ of the volume you need a subcube of side $r^{1/n}$: to capture 10% of the data you need 30% of each coordinate range in 2D, but **80%** in 10D. The other way round: side $0.3$ captures $\sim30\%$ of the volume in 1D, $9\%$ in 2D, $2.7\%$ in 3D and $\sim0.0006\%$ in 10D.
2. **Low sampling density.** Density is proportional to $l^{1/n}$: if 100 points suffice in $\mathbb{R}^1$, you need $100^{10}$ for comparable accuracy in $\mathbb{R}^{10}$.
3. **Irrelevant features.** If the target depends on 2 out of 20 features, the "similarity" is dominated by the 18 irrelevant ones. Fix: feature weighting (stretch the axes) or [[Feature Selection]].


Also relevant for [[Linear Basis Expansion]]: large feature expansions raise the overfitting risk.

> [!warning]
> [[SVM]] does **not** solve it. SVM copes well with a high-dimensional **feature** space (the complexity is ruled by the margin, not by the dimension), but the curse lives in the **input** space, and that is untouched.

## Controls

- A different **bias**: assuming the data come from a composition of factors lets a model distinguish exponentially many regions with $O(k)$ examples, i.e. generalize non-locally $\to$ [[Deep Learning]], [[Distributed Representation]].
- [[Feature Selection]] and dimensionality reduction.
- Feature weighting (stretching the axes).
- [[Regularization]].

These matter in terms of ML because they directly affect the **generalization** capability.
