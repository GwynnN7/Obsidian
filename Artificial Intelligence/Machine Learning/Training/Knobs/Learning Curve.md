---
tags: [machine-learning, training]
---

# Learning Curve

The plot of the error against the training iterations/epochs.

> [!important]
> **(!) Always look at the learning curve while training a NN.** It is the cheapest diagnostic there is, and it tells you which approach to keep in the early phases of the design.

## What to look for (optimization quality, on TR)

| Shape | Meaning |
|---|---|
| diverging | $\eta$ far too high |
| flat / stuck | $\eta$ far too low |
| unsmooth, irregular | too small a batch, high $\eta$, ... **(!) avoid it** |
| smooth, decreasing, then flattening | what you want |

The **absolute value** of the training error depends on the model capacity (number of units, $\lambda$, ...) and on the other hyperparameters - here we are only judging quality and speed of convergence.

## What to look for (generalization)

Plotting TR **and** VL/TS error together gives the typical backprop behaviour: both go down, then the validation curve turns up while the training one keeps going down $\to$ **overtrained**. The good zone to stop is around the minimum of the validation curve.

This is the empirical counterpart of the VC-bound plot: since the effective number of parameters grows during training, halting the training limits the effective complexity. See [[Early Stopping]] and [[Structural Risk Minimization]].

> [!note]
> A well-regularized model shows the VL curve **following** the TR curve, so no ES is needed. Careful though: the two curves can also be close because a too high $\lambda$ is causing underfitting.

Related: [[Learning Rate]], [[Stopping Criteria]], [[Hyperparameter Guidelines]].
