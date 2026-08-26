---
tags: [machine-learning, theory]
---

# Cover's Theorem

> "A complex pattern-classification problem, cast in a high-dimensional space nonlinearly, is more likely to be linearly separable than in a low-dimensional space, provided that the space is not densely populated."


Extreme case, with a deterministic mapping: given $l$ examples, lift them onto the vertices of the $(l-1)$-dimensional simplex. Then **every** binary partition of the samples is linearly separable.

## Why it matters

It is the theoretical justification for the whole "expand, then separate linearly" family:

- [[Linear Basis Expansion]] - fixed $\phi$;
- [[Neural Networks]] - adaptive $\phi$ learnt from data;
- [[Randomized Neural Networks]] - **random** $\phi$, with many units;
- [[Echo State Networks]] - random $\phi$ over sequences;
- [[SVM]] and [[Kernel Methods]] - the same property exploited by **kernels**, which get the high-dimensional space without ever building it.

The caveat is in the last clause: *provided that the space is not densely populated*. Expanding the dimension without enough data is exactly the [[Curse of Dimensionality]], and separability bought this way still has to be paid for with complexity control - see [[VC Dimension]] and [[Regularization]].

Related: [[Statistical Learning Theory]], [[Classification]].
