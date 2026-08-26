---
tags: [machine-learning, training]
---

# Batch, Online and Mini-Batch

**Epoch** = one entire cycle of presentation of all the training patterns.

## Batch

Sum the gradients of all $l$ patterns over an epoch, **then** update:

$$
-\frac{\partial E(w)}{\partial w_{tu}}=-\sum_{p=1}^{l}\frac{\partial E_p(w)}{\partial w_{tu}}=\sum_{p=1}^{l}\delta_{p,t}\,o_{p,u}.
$$

- More accurate estimation of the gradient.
- Slower: you wait for all the data.

## On-line / stochastic (SGD)

Update after **each** pattern $p$, without waiting for the sum:

$$
-\frac{\partial E_p(w)}{\partial w_{tu}}=\delta_{p,t}\,o_{p,u}.
$$

- Makes progress with each example it looks at: can be faster, but needs a **smaller** $\eta$.
- The gradient for a single point is a noisy approximation of the total one $\to$ zig-zag descent.
- Can help escaping local minima; adds noise, which has a regularization effect.
- Cannot exploit parallel processing of examples; may be unstable.
- **Shuffle the patterns in each epoch** - otherwise a fixed order introduces a bias/drift in the descent.

## Mini-batch

Split the epoch into parts: sum the gradient over $mb$ patterns, update, repeat until all the training data are used.

$$
1 < mb < l
$$

- Compromise between the two; $mb$ compatible with e.g. GPU memory.
- Use random sampling (shuffle at the beginning, or per mini-batch) so the estimate is unbiased.
- With very large TR sets or online streaming: **no epochs at all**.
- Key point for scaling: the cost of each SGD update does not depend on $l$, only on $mb$ - this is what extends learning to millions of examples.

## Interaction with $\eta$

See [[Learning Rate]] - dividing the gradient by $l$ (LMS) or by $mb$ changes the effective $\eta$, and comparing batch with on-line at the same $\eta$ is meaningless.

Related: [[Gradient Descent]], [[Training]], [[Momentum]].
