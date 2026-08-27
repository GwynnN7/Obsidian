---
tags: [machine-learning, models, theory]
---

# Bayes Optimal Classifier

If we knew the density $P(x,y)$, the best possible rule is to output the most probable class:

$$
h(x)=\arg\max_v P(v\mid x),\qquad v\in\{C_1,\dots,C_K\}.
$$

- Its error rate is the **Bayes rate**: the minimum achievable error rate given the distribution of the data.
- It is a reference point, not an algorithm - the generating density is unknown in practice.

[[k-NN]] approximates it by:

- replacing the conditional probability at a point with the conditional probability in a **neighborhood** of the point,
- estimating the probabilities with training-sample proportions.

Related: [[Classification]], [[Generalization and Risk]], and [[Probability and Decision Theory]] / [[Bayesian Networks]] for the same rule in its own right.
