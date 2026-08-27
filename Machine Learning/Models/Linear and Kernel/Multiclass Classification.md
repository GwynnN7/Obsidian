---
tags: [machine-learning, models]
---

# Multiclass Classification

Two simple ways to go from binary to $K$ classes with linear-type models:

## 1-of-K representation

`{red, green, blue}` $\to$ $(0,0,1),(0,1,0),(1,0,0)$: solve $K$ models, one per output component. Same as multi-output [[Neural Networks]]: the winner is the output unit with the highest value.

## OVA - one-vs-all

- Train $K$ binary classifiers, each separating one class from all the others.
- To classify a new example run all $K$ and take the one with the largest (most positive) output.

## AVA - all-vs-all (one-vs-one)

- One classifier per **pair** of classes: $K(K-1)$ of them.
- The winner is the class with the max sum of outputs against all the others, or the class with most votes.
- Each classifier has a much smaller training set.

## Problems

- **Masking:** with high $K$ some classes can be masked by others.
- More sophisticated approaches exist; some models handle multi-output directly.

## In NN

- Sigmoidal (tanh) outputs with $0/1$ or $-1/+1$ encoding, or
- **softmax** for $0/1$ targets, so the outputs sum to 1 and can be read as $p(\text{class}=k\mid x)$:

  $$
  o_k(x)=\frac{e^{net_k}}{\sum_{j=1}^{K}e^{net_j}} .
  $$

  Popular nowadays for classification; then the natural loss is **cross-entropy** instead of MSE, e.g. for one unit

  $$
  -\sum_{i\in TR}\{d_i\log(out(x_i))+(1-d_i)\log(1-out(x_i))\}.
  $$

  Loss $=0$ $\to$ no classification errors.

Related: [[Classification]], [[Activation Functions]], [[Linear Models]].
