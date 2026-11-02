---
tags: [machine-learning, training]
---

# Stopping Criteria

## Based on the error

- Mean error $<\epsilon$ - the best option **if** you know the tolerance of the data (expert knowledge), which we usually don't.
- Max (instead of mean) tolerance on the data.
- Classification: number of misclassified patterns / error rate.

## Internal criteria

- No more relevant weight changes.
- Near-zero gradient (Euclidean norm $<\epsilon$).
- No significant error decrease over an epoch (e.g. less than 0.1%).

> [!warning]
> These may be **premature**, e.g. if $\eta$ is too small. Apply them over $k$ epochs - this is the **patience**.

## Fixed number of epochs

- (!) Always stop after an excessive number of epochs, just to escape a too slow convergence.
- **But never stop only because a preselected number of epochs has elapsed:**
  - too small $\to$ underfitting, or an unfair advantage to a high $\eta$ (which needs fewer epochs);
  - too large $\to$ overfitting or wasted time;
  - one number cannot hold for every hyperparameter configuration nor for every fold of a [[Cross Validation]];
  - and what number would you use for the final retraining?
- Even *selecting* the number of epochs by model selection is not the best practice: a bit better than a fixed number, but still not accurate, given the CV issue and the fact that we have plenty of other criteria.

## The two real options

1. **Well-regularized model + convergence criterion** - stop at training convergence (negligible error decrease, tiny gradient). Preferred when [[Regularization]] is doing its job.
2. **[[Early Stopping]]** - use a validation set. Empirical, part of model selection, spends part of the data.

You can also use both: with good regularization, ES simply never fires.

Related: [[Training]], [[Learning Curve]], [[Validation Pitfalls]].
