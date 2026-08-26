---
tags: [machine-learning, validation]
---

# Model Selection and Assessment

## Two different goals

- **Model selection:** compare candidate models and/or hyperparameter configurations (polynomial order, #units, $\lambda$, $\eta$, ...) and choose the best one. **It returns a model.**
- **Model assessment:** estimate the prediction error / risk of the ultimately chosen model on new data. **It returns an estimation value.**

$\Lambda$ denotes the hyperparameter configuration - but think of it as also representing *different models*, since selecting among models needs the same machinery.

## Identify your goal first

1. "Give me the best model I can get from this data set" $\to$ model selection.
2. "Give me the expected performance of a model obtained by ERM on this data set" $\to$ model assessment.
3. "Both" $\to$ do both, merged (see [[Validation Schemas]]).

Note: the single training set we are given is used by us for all three purposes.

## Holdout roles

If the data set is large enough, e.g. 50% TR, 25% VL, 25% TS, all **disjoint**:

| Set | Purpose |
|---|---|
| $D_{TR}$ | train/fit the parameters (minimize $R_{emp}$) - **training** |
| $D_{VL}$ | select the model and the hyperparameters - **model selection** |
| $D_{TS}$ | final risk estimate (estimate $R$) - **model assessment** |

$D_{TR}\cup D_{VL}$ is the **development / design set**: the data used to build the final model (train the candidates and select the best one).

> [!warning] Test-set leakage
> If test results influence any design decision, you are doing model selection on that data and the test estimate becomes optimistic. **The test set must be separated in advance**, before any feature or model selection.
>
> - Blind test set (as in ML competitions) makes this impossible by construction.
> - Like a test with the answer key next to it: if you have seen the solutions, it is not a test.

## How much data is enough?

Depends on the model complexity and on the signal-to-noise ratio of the underlying function (few data are enough for a linear model on a linearly separable, noiseless task). When the data are not enough for three significant splits $\to$ [[Cross Validation]].

## [[Hyperparameter Search]]

- [[Grid Search]]: coarse/wide search, then fine search around the promising regions.
- [[Random Search]]: avoids the exhaustive Cartesian cost, especially with many hyperparameters.
- Sequential tuning introduces bias: a value selected for one architecture may not suit another. Search on a grid with **all** the combinations of the significant hyperparameters.

## Related

- [[Validation Schemas]] - the algorithms, including the retraining step.
- [[Validation Pitfalls]] - what goes wrong in practice.
- [[Cross Validation]], [[Early Stopping]], [[Statistical Learning Theory]].
