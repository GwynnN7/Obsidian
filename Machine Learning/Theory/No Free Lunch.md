---
tags: [machine-learning, theory]
---

# No Free Lunch

There is no universally best learning method. If an algorithm does better than others on some problems, it must pay with worse results on other problems.

(Devroye 1982; Wolpert & Macready 1997.)

## But models are not all equivalent

What actually separates them:

- **Flexibility** - can the model, in principle, approximate arbitrary functions? (linear vs [[Neural Networks]])
- **Control of the complexity** - can that flexibility be reined in? ([[Regularization]], [[Structural Risk Minimization]])

Flexible models + principled complexity control is the core of ML.

Related: [[Inductive Bias]], [[Bias-Variance Tradeoff]].
