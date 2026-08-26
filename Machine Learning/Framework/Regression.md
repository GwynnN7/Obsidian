---
tags: [machine-learning, tasks]
---

# Regression

Estimate a real-valued function from a finite set of **noisy** samples: known pairs $(x, f(x)+\text{noise})$, target $y\in\mathbb{R}$ (or $\mathbb{R}^K$ for multi-output).

- Inner loss: squared error $(h(x)-y)^2$, see [[Loss Functions]].
- Metrics: [[Mean Squared Error]]/RMSE, MAE, max error, $R$ - see [[Regression Metrics]].
- Output units for NN: **linear**, one per quantitative response.

## Models in these notes

[[Linear Models]] (LMS), [[Linear Basis Expansion]] / polynomial regression, [[k-NN]] (mean of the neighbors), [[Neural Networks]], [[SVM Regression]] ($\epsilon$-insensitive).

> [!note]
> The polynomial-fitting example (target $\sin(2\pi x)$ + gaussian noise, degree $M$) is the standard illustration of [[Underfitting]] / [[Overfitting]] and of the effect of $\lambda$ - see [[Regularization]] and [[Statistical Learning Theory]].

Related: [[Supervised Learning]].
