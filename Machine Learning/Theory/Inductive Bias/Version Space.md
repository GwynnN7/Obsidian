---
tags: [machine-learning, theory]
---

# Version Space

A hypothesis $h$ is **consistent** with the training set if $h(x)=d(x)$ for every example $(x,d(x))$ in TR.

The version space is the subset of $H$ consistent with all the examples:

$$
VS(H,D)=\{h\in H\mid h(x)=d(x)\ \forall(x,d)\in D\}.
$$

- With conjunctive rules the whole VS can be found efficiently, without enumerating $H$.
- With an unbiased $H$ (the power set of $X$) the VS classifies unambiguously **only** the training examples - see [[Concept Learning]].

Related: [[Hypothesis Space and Inductive Bias]].
