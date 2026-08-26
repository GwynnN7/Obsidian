---
tags: [machine-learning, theory]
---

# Underfitting

The model capacity is too limited to capture the structure, so training **and** test error are both high.

- SLT view: $R_{emp}$ stays high, so the bound on $R$ is high even with a small VC-confidence. See [[Statistical Learning Theory]].
- Bias-variance view: high bias, low variance - the model is too rigid. See [[Bias-Variance Tradeoff]].
- Typical causes: too few hidden units, too low polynomial degree, $\lambda$ too high, stopping training too early.

Related: [[Overfitting]], [[Regularization]].
