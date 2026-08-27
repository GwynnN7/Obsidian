---
tags: [machine-learning, ensembles]
---

# Bagging

**B**ootstrap **agg**regat**ing**: train $K$ models on different [[Bootstrap]] samples (resampling with replacement) of the training set, then combine them.

- Regression: mean prediction.
- Classification: vote of the $K$ classifiers.

## Why it works

- Averaging reduces **variance**, so it helps most with unstable / high-variance, low-bias models.
- Think of the "low $\lambda$" picture in [[Bias-Variance Tradeoff]]: single fits are wild, the mean $\bar h(x)$ is good.
- The average of $K$ separation curves is likely to end up "in the middle", which increases the margin of the classifier. *In medio stat virtus.*

Related: [[Ensemble Learning]], [[Boosting]].
