---
tags: [machine-learning, theory, models]
---

# Curse of Dimensionality

As the number of input dimensions grows, the volume of the problem space grows so fast that the available data become sparse.

Three manifestations (details and numbers in [[k-NN]]):

1. Truly close neighbors are hard to find: neighborhoods must become spatially large and stop being local.
2. Sampling density is proportional to $l^{1/n}$, so the number of examples needed for the same accuracy explodes.
3. Irrelevant features dominate the similarity measure.

Also relevant for [[Linear Basis Expansion]]: large feature expansions raise the overfitting risk.

> [!warning]
> [[SVM]] does **not** solve it. SVM copes well with a high-dimensional **feature** space (the complexity is ruled by the margin, not by the dimension), but the curse lives in the **input** space, and that is untouched.

## Controls

- [[Feature Selection]] and dimensionality reduction.
- Feature weighting (stretching the axes).
- [[Regularization]].

These matter in terms of ML because they directly affect the **generalization** capability.
