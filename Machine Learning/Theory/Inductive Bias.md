---
tags: [machine-learning, theory]
---

# Inductive Bias

The assumptions a learner uses to generalize beyond the observed data.

- **Language bias:** constraints on the hypothesis space $H$.
- **Search bias:** constraints/preferences in the learning/search algorithm.

Two facts to keep together:

1. Bias is not only an efficiency trick - it is **needed** for generalization ([[Concept Learning]]).
2. Every state-of-the-art ML approach has one; the useful question is *which* bias a model has.

Full discussion in [[Hypothesis Space and Inductive Bias]].
