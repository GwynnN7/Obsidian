---
tags: [machine-learning, moc, theory]
---

# Theory

Why learning from finite data can work at all, and what "complexity control" means.

> [!note] One section per subfolder
> The three sections match the folders inside `Theory/`.

## Inductive Bias

What we can represent, and the assumptions without which nothing generalizes.

- [[Hypothesis Space and Inductive Bias]] - the full discussion.
- [[Inductive Bias]] - language bias vs search bias, in short.
- [[Version Space]] - the hypotheses consistent with the data.
- [[Concept Learning]] - the discrete-$H$ playground where the bias argument is **proved**.
- [[Learning Algorithm]] - the search through $H$.
- [[No Free Lunch]] - there is no universally best learner.

## Generalization

What can go wrong, and how it is measured.

- [[Generalization and Risk]] - loss, empirical risk, true risk, ERM.
- [[Overfitting]] / [[Underfitting]] - the two failure modes.
- [[Bias-Variance Tradeoff]] - decomposition of the expected error into bias, variance and noise, with the derivation.
- [[Bayes Optimal Classifier]] - the best any learner could do, and the error floor it sets.

## Capacity and Bounds

The analytical side: bound the true risk instead of measuring it.

- [[Statistical Learning Theory]] - the VC-bound, and the other analytical criteria (AIC, BIC, MDL).
- [[VC Dimension]] - the capacity measure it uses.
- [[Structural Risk Minimization]] - how to use the bound to choose a model.
- [[Cover's Theorem]] - why casting a problem into a higher-dimensional space helps.

> [!important]
> [[Statistical Learning Theory]] and [[Bias-Variance Tradeoff]] say the same thing in two languages: **fit the data, but not too much**.

## Related

- [[Ensemble Learning]] - combining models; it lives in [[Models]] because a committee *is* a model.
- [[Curse of Dimensionality]] - what goes wrong when $n$ grows.
- [[Regularization]] - the practical way to control complexity.
- [[Validation]] - the empirical way to find the trade-off.
