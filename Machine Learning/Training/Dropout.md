---
tags: [machine-learning, training]
---

# Dropout

A [[Regularization|regularization]] technique for neural networks: during **training**, randomly drop (set to zero) each unit with probability $p$, independently at every forward pass.

$$
\text{keep each unit with probability } 1-p
$$

- The dropped units take no part in the forward pass and get no gradient in the backward pass.
- At **test** time all the units are used - no dropping.

## Getting the scale right

If we drop units during training, the expected input to the next layer is smaller than at test time. Two equivalent fixes:

- **Classic:** at test time multiply the outgoing weights by $(1-p)$.
- **Inverted dropout** (what libraries do): during training divide the surviving activations by $(1-p)$, so test time needs no change at all.

> [!warning]
> Dropout must be **off** at inference. Forgetting to switch it off is a classic source of "my test results are worse and noisy".

## Why it works

- **Prevents co-adaptation:** a unit cannot rely on the presence of a specific other unit, so it has to learn features that are useful on their own.
- **Implicit ensemble:** each mask defines a different "thinned" network; training with dropout approximates training an exponential number of networks that **share weights**, and testing with all units approximates averaging them. See [[Ensemble Learning]].
- It is a form of noise injection, so it belongs with data augmentation in the family of "regularization by noise" - compare the noise of on-line SGD in [[Batch Online and Mini-Batch]] and the randomness catalogue in [[Randomized Neural Networks]].

## In practice

- Typical: $p\approx0.5$ on the fully-connected hidden layers, lower ($0.1$-$0.2$) on the inputs.
- Most useful on the large **fully-connected** layers, which hold the bulk of the parameters - see [[CNN]].
- Convolutional layers are already regularized by weight sharing, so they usually need less (or a spatial variant).
- $p$ is a hyperparameter $\to$ [[Grid Search]], like $\lambda$.
- With strong dropout the training error goes up and gets noisier: that is expected, judge on validation.

> [!note]
> Dropout does the same *job* as the weight penalty of [[Regularization]] - controlling capacity - by a completely different mechanism. They can be combined, and both are selected on a validation set.

Related: [[Regularization]], [[Batch Normalization]], [[Neural Networks]], [[Deep Learning]], [[Hyperparameter Guidelines]].
