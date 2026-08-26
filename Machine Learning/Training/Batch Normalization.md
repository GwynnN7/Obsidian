---
tags: [machine-learning, training]
---

# Batch Normalization

Normalizing the **inputs** is standard practice ([[Data and Preprocessing]]). Batch normalization does the same thing to the data flowing **between** the layers, so it stays normalized all the way down.

For each layer, take the matrix $[\text{mini-batch}\times\text{unit activations}]$ and normalize each activation with the mean and variance computed **on that batch** - shifting the values to zero mean and unit variance (a reparametrization). Two learned parameters per unit restore scale and shift, and the whole thing is included in [[Backpropagation]].

## Why it helps

- **Faster learning and higher accuracy** on deep models. It is what makes deep networks with sigmoidal units and [[CNN]] trainable efficiently.
- Keeps units away from the saturated zone, so the gradient does not shrink layer after layer - see the vanishing gradient discussion in [[Deep Learning]].
- **Regularization effect**, from the noise added at training time by the choice of the mini-batch: the statistics of each unit depend on which patterns happen to be in the batch. Same family as [[Dropout]] and the noise of on-line SGD.

## In practice

- Depends on the mini-batch, so the batch size matters $\to$ [[Batch Online and Mini-Batch]]. Very small batches give noisy statistics.
- At **inference** the batch statistics are replaced by running averages accumulated during training - so, as for [[Dropout]], training mode and inference mode are different.

Related: [[Regularization]], [[Deep Learning]], [[Weight Initialization]], [[Optimizers]].
