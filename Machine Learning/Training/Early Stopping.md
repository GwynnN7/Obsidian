---
tags: [machine-learning, training, validation]
---

# Early Stopping

## What it is

Stop training before fully minimizing the training error, using evidence that further fitting no longer improves generalization.

- An **empirical complexity-control** method: since the effective number of parameters (and the VC-dim) grows during training, halting limits the effective complexity. See [[Neural Networks]] and [[Structural Risk Minimization]].
- It is a **model-selection decision**: it needs validation evidence, never the final test set.
- It complements [[Regularization]]; it does not replace correct validation.
- It is tricky because it uses part of the data just to decide *when* to stop, without giving a concrete value of a hyperparameter - and it gets complicated inside CV.

## Why a fixed epoch count is wrong

See [[Stopping Criteria]] - too small underfits (and unfairly favours a large $\eta$), too large overfits or wastes time, and one number cannot fit every configuration nor every fold.

## Basic holdout procedure

1. Reserve $D_{VL}$ from $D_{TR}$.
2. Train on $D_{TR}$ and monitor the validation error on $D_{VL}$.
3. Keep the parameters of the best validation point - with a **patience** rule (wait more than one epoch before deciding), not one noisy increase. A backtracking approach works too.
4. Use this to select the stopping strategy/configuration.

> [!warning] No test-set stopping
> Selecting the stopping point on the test error turns the test set into a validation set, and its error is no longer an unbiased risk estimate. See [[Validation Pitfalls]].

## Within cross-validation

- Activate ES **independently inside every training/validation fold**, together with the other hyperparameters.
- Each fold can have a different best stopping point, because its training/validation data differ.
- The CV result is used for [[Model Selection and Assessment|model selection]], never as a reason to touch the final test set.

## Final retraining after CV

There is no perfect single epoch count coming out of CV:

- **Possible but imperfect:** treat #epochs as a hyperparameter and average the validation-optimal epochs over the folds. Issue: the final training set has a different size, so its best stopping point can move - risking undertraining or overtraining.
- **Better:** take the **training error** at each fold's validation-optimal point, average that value, and retrain until the same level of fitting is reached. This avoids the undertraining effect.
- **Alternative:** keep a VL set during the final retraining too, i.e. use a VL set any time you train the model.

## Relation to regularization

- ES sacrifices validation data to decide when to stop.
- [[Regularization]] adds a penalty to the objective and controls the capacity directly; a well-regularized model can then use internal convergence criteria (negligible training error decrease, near-zero gradient).
- Your choice - and regularization anyway reduces the need for a strict dependence on ES.

## Checklist

- Train on $D_{TR}$.
- Select the stopping point using $D_{VL}$ / inner CV **only**.
- Assess the final procedure once on a separate $D_{TS}$.

Related: [[Cross Validation]], [[Learning Curve]], [[Model Selection and Assessment]].
