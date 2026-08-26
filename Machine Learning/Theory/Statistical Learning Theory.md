---
tags: [machine-learning, theory]
---

# Statistical Learning Theory

## Purpose

A formal framework that bounds the true risk $R$ starting from the empirical risk, the number of data, the capacity of $H$ and a confidence level - **independently of the specific model or learning algorithm**.

## Approximating the estimation step

Two families of approaches to estimate the error:

- **Analytical:** AIC/BIC (limited to models linear in the parameters), MDL, and **SRM + VC-dim** (this note).
- **By resampling:** [[Cross Validation]], [[Bootstrap]]. See [[Validation]].

## The VC bound

It holds with probability $1-\delta$:

$$
R(h) \le \underbrace{R_{emp}(h)}_{\text{training error}} + \underbrace{\epsilon(1/l,\ VC(H),\ 1/\delta)}_{\text{VC-confidence}}
$$

Reading it:

- $\epsilon$ **decreases** as the number of examples $l$ grows.
- $\epsilon$ **grows** with the VC dimension.
- $\delta$ is the confidence ($\delta=0.01$ $\to$ the bound holds with probability $0.99$).
- $R_{emp}$ decreases with more complex models.

So:

- Too simple model (low VC): $R_{emp}$ high $\to$ [[Underfitting]].
- Too complex model (high VC, $l$ fixed): $R_{emp}$ low but VC-confidence high $\to$ $R$ can grow $\to$ [[Overfitting]].
- The sum has a **U shape**: the minimum is the best trade-off.

Different formulations exist for different classes of functions and tasks, e.g. $R\le R_{emp}+\epsilon(VC/l,-\ln(\delta/l))$.

> [!note]
> The VC-confidence can be computed **before** learning, and the bound holds for all $h(x,\theta)$ at once, so we can search over the whole class.

## How many examples?

For many reasonable classes (e.g. linear approximators) the VC-dim is linear in the number of free parameters. Consequence: to learn well we need a number of examples **linear in the VC dimension**.

## [[Structural Risk Minimization]]

Choose among nested hypothesis spaces $H_1\subset H_2\subset\dots$ by minimizing the bound rather than the training error.

## Use and limits of the bound

Good for:

- a fundamental theoretical ground for principled ML, independent of model details;
- highlighting the role of complexity control;
- model selection/comparison, where only the **relative** size/ranking of the error matters;
- driving the development of new models (this is where [[SVM]] comes from).

Bad for:

- estimating the predictive error in practice - rarely used. The bounds are **worst case** and can be very loose (the VC-confidence can be hundreds of times bigger than the actual overfitting effect);
- classes whose VC-dim is hard to compute.

## Related

- [[VC Dimension]]
- [[Regularization]] - Tikhonov as a practical SRM
- [[Bias-Variance Tradeoff]] - the same trade-off, other language
- [[Model Selection and Assessment]]
