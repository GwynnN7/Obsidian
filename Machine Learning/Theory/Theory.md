---
tags: [machine-learning, moc, theory]
---

# Theory

Why learning from finite data can work at all, and what "complexity control" means.

## The setup

- [[Hypothesis Space and Inductive Bias]] - what we can represent, and the assumptions that make generalization possible.
- [[Inductive Bias]] - language bias vs search bias.
- [[Version Space]] - the hypotheses consistent with the data.
- [[Concept Learning]] - the discrete-$H$ playground where the bias argument is proved.
- [[Learning Algorithm]] - the search through $H$.
- [[No Free Lunch]] - there is no universally best learner.
- [[Cover's Theorem]] - why casting a problem into a higher-dimensional space helps.

## Generalization

- [[Generalization and Risk]] - loss, empirical risk, true risk, ERM.
- [[Overfitting]] / [[Underfitting]] - the two failure modes.

## The two frameworks that explain the trade-off

- [[Statistical Learning Theory]] - analytical bound on $R$: [[VC Dimension]], VC-bound, [[Structural Risk Minimization]].
- [[Bias-Variance Tradeoff]] - decomposition of the expected error into bias, variance and noise.

Both say the same thing in different languages: **fit the data, but not too much**.

## Ensembles

- [[Ensemble Learning]] - why combining models helps.
- [[Bagging]] - reduces variance.
- [[Boosting]] - reduces bias.

## Related

- [[Curse of Dimensionality]] - what goes wrong when $n$ grows.
- [[Regularization]] - the practical way to control complexity.
- [[Validation]] - the empirical way to find the trade-off.
