---
tags: [machine-learning, validation]
---

# Hyperparameter Search

**Hyperparameters** = the values you have to set up to run the training, i.e. parameters not directly learnt by the estimator: learning rate $\eta$, regularization $\lambda$, momentum $\alpha$, batch size, number of hidden units, polynomial order, $k$ in k-NN, ...

They are searched in a hyperparameter space, of course **by model selection on a validation set**.

## Search approaches

- **[[Grid Search]]:** exhaustive over a grid of values. Coarse/wide first, then fine around the promising regions.
- **[[Random Search]]:** sample configurations, fixing the budget independently of the number of hyperparameters.
- **Automatic:** Bayesian approaches, [[Evolutionary Algorithms|evolutionary]] approaches, ...

> [!note]
> Don't use the automatic ones uncritically. Compare with the grid search, which teaches you more on a first approach to ML.

## Correct use

- Select configurations using validation data or an inner [[Cross Validation]] loop.
- **Do not tune sequentially** without rechecking the combinations: hyperparameters interact (e.g. $\eta$ and #hidden units). See [[Validation Pitfalls]].
- Do not use the final test set for tuning.
- Some values can be fixed in a preliminary experimental phase if they show to be irrelevant - but be careful about the sequential-selection bias.

## Which ones matter

The ones related to the VC-dim: number of units, $\lambda$, degree, ... See [[Structural Risk Minimization]] and [[Hyperparameter Guidelines]].

See [[Model Selection and Assessment]].
