---
tags: [machine-learning, theory]
---

# Learning Algorithm

A (heuristic) search through the [[Hypothesis Space and Inductive Bias|hypothesis space]] $H$ for the best approximation of the unknown target, typically the $h$ with minimum error on the training data.

- It fits the free parameters of the model to the task: best $w$ for [[Linear Models]] and [[Neural Networks]], best rules for symbolic models, ...
- $H$ is usually too large for exhaustive search $\to$ the search is local/heuristic $\to$ it carries a **search bias**.

Other names for the same thing, depending on the field: inference (statistics/logic), adapting (biology), optimizing (math), training (NN), function approximation (math).

Related: [[Gradient Descent]], [[No Free Lunch]].
