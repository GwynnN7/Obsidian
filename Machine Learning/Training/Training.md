---
tags: [machine-learning, moc, training]
---

# Training

How the weights are actually found, and how the search is kept under control.

## The basic algorithm

[[Gradient Descent]]: start from small random weights, move against the gradient of the loss, repeat.

1. Fix $w_{initial}$ (small) and $\eta$ with $0<\eta<1$.
2. Compute $\Delta w=-\partial E(w)/\partial w$.
3. $w_{new}=w+\eta\,\Delta w$.
4. Repeat until convergence or $E(w)$ small enough.

For an MLP, step 2 is done by [[Backpropagation]]: a forward pass to compute the outputs, then a backward pass to retro-propagate the deltas.

> [!note] Nice interpretation
> Backprop is a **path through the loss/weight space**. The path depends on: the data, the model, the starting point ([[Weight Initialization]]), the rate of convergence ([[Learning Rate]], [[Momentum]]), and the final point ([[Stopping Criteria]]). Together they are a control over the search in $H$.

> [!note] One section per subfolder
> The three sections match the folders inside `Training/`.

## Algorithm

How the gradient is computed and applied.

- [[Gradient Descent]] - the algorithm itself.
- [[Delta Rule]] - the error-correction update.
- [[Backpropagation]] - the gradient for a multilayer network.
- [[Batch Online and Mini-Batch]] - when to update.
- [[Optimizers]] - SGD, RMSProp, Adam, AdamW.

## Knobs

Everything that decides *the path* rather than the destination.

- [[Learning Rate]] - how big the step is.
- [[Momentum]] - inertia, and Nesterov.
- [[Weight Initialization]] - where to start.
- [[Multiple Minima]] - non-convex loss, and why it is not a disaster.
- [[Stopping Criteria]] - when to stop.
- [[Learning Curve]] - the plot that tells you if it is working.
- Gradient **clipping** and the vanishing/exploding gradient across layers $\to$ [[Deep Learning]].

## Complexity Control

Everything that decides *how complex* the final model is allowed to be.

- [[Regularization]] - penalty on the loss. The main tool.
- [[Dropout]] - regularization by randomly dropping units.
- [[Batch Normalization]] - normalize the activations between layers; regularizes as a side effect.
- [[Early Stopping]] - empirical, uses validation data.
- Number of hidden units, [[Cascade Correlation]], pruning.

> [!important]
> We do **not** want the global minimizer of $R_{emp}(w)$ - that is likely an overfitting solution. This is the difference between ML and pure optimization: optimizing on the training data is not our purpose. Validation matters more than the optimization technique.

## Related

- [[Hyperparameter Guidelines]] - the practical do / don't page.
- [[Neural Networks]], [[Validation]].
