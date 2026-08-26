---
tags: [machine-learning, moc, validation]
---

# Validation

> Evaluation of performance for a ML system = evaluation of the **generalization / predictive accuracy**. Validation, validation, validation.
>
> Easy use of ML tools versus **correct** use of ML.

## The two aims (the thing to never forget)

- **Model selection:** estimate the generalization error of different models / hyperparameter configurations in order to choose the best one. *It returns a model.*
- **Model assessment:** having chosen a final model (or class of models), estimate its prediction error on new test data. *It returns an estimation value.*

**Gold rule:** keep the goals separate and use separate data sets in each phase.

Details in [[Model Selection and Assessment]].

## Two gold rules (repetita)

1. TR result is not a good estimate of VL and TS results.
2. VL result is not a good estimate of the TS result.

Hence: don't use VL results as a risk estimate, and don't use TS results for model selection **in any form**. Or, in one rule: keep separation between goals and use separate sets in each phase.

## How to estimate

- **Analytically:** AIC/BIC (limited to models linear in the parameters), MDL, SRM + VC-dim $\to$ [[Statistical Learning Theory]].
- **By resampling:** [[Cross Validation]] (hold-out, K-fold, ...), [[Bootstrap]]. This is what we do in practice.

## The notes

- [[Model Selection and Assessment]] - the two aims, the data splits, the roles.
- [[Validation Schemas]] - the meta-algorithms, written out.
- [[Cross Validation]] - K-fold, LOO, variants, nested CV.
- [[Validation Pitfalls]] - the counterexample and the FAQ / typical errors.
- [[Hyperparameter Search]] - [[Grid Search]], [[Random Search]].
- [[Stratified Sampling]], [[Bootstrap]] - how to split/resample.
- [[Feature Selection]] - and why it belongs **inside** the selection process.

## Metrics

[[Classification Metrics]], [[Confusion Matrix]], [[ROC Curve]], [[Regression Metrics]], [[Mean Squared Error]].

> [!note]
> These schemas are a very useful guide, but **not a recipe book**: the point is to rationalize the problem, not to apply a ritual. Then be reasonable - most of the running time depends on the validation choices, so avoid both a too rough evaluation and a never-ending one. When in doubt, use them: better than fantasy.
