---
tags: [machine-learning, theory]
---

# Statistical Learning Theory

## Purpose

A formal framework that bounds the true risk $R$ starting from the empirical risk, the number of data, the capacity of $H$ and a confidence level - **independently of the specific model or learning algorithm**.

## Approximating the estimation step

Given a hyperparameter (implicit or explicit) that moves the model complexity, we want the value that minimizes the error on future data. That error has to be *estimated*, and there are two families of ways to do it:

- **Analytically:** SRM + VC-dim (this note), and the penalized-likelihood criteria AIC / BIC, and MDL - see *Other analytical criteria* below.
- **By resampling:** direct estimation of the extra-sample error via [[Cross Validation]] and [[Bootstrap]]. See [[Validation]].

## The VC bound

It holds with probability $1-\delta$:

$$
R(h) \le \underbrace{R_{emp}(h)}_{\text{training error}} + \underbrace{\epsilon(1/l,\ VC(H),\ 1/\delta)}_{\text{VC-confidence}}
$$

The right-hand side is the **guaranteed risk**. Reading it:

- $\epsilon$ **decreases** as the number of examples $l$ grows ($l$ and $\delta$ sit in the denominator), and the bound gets close to $R$.
- $\epsilon$ **grows** with the VC dimension.
- $\delta$ is the confidence ($\delta=0.01$ $\to$ the bound holds with probability $0.99$).
- $R_{emp}$ decreases with more complex models.

So:

- Too simple model (low VC): $R_{emp}$ high $\to$ [[Underfitting]].
- Too complex model (high VC, $l$ fixed): $R_{emp}$ low but VC-confidence high $\to$ $R$ can grow $\to$ [[Overfitting]].
- The sum has a **U shape**: the minimum is the best trade-off.

### Explicit form, 0/1 loss

Writing $N$ for the number of data (the $l$ used elsewhere), for the 0/1 loss the VC-confidence is

$$
\epsilon(VC,N,\delta)=\sqrt{\frac{VC\left(\ln\frac{2N}{VC}+1\right)-\ln\frac{\delta}{4}}{N}}
$$

with probability at least $1-\delta$, for every $VC<N$.

The shape confirms the reading above: $\epsilon\to0$ as $N$ grows, and $\epsilon$ grows with $VC$. The condition $VC<N$ is the formal version of "you need more data than capacity" - past that point the bound says nothing.

### Other formulations

Different bounds exist for different classes of functions and different tasks, e.g.

$$
R\le R_{emp}+\epsilon\!\left(VC/l,\ -\ln(\delta/l)\right).
$$

They all have the same two ingredients and the same U shape; only the exact function $\epsilon$ changes.

> [!note]
> The VC-confidence can be computed **before** learning - it needs only $VC$, $N$ and $\delta$, not the trained model. And the bound holds for all $h(\boldsymbol{x},\theta)$ in the class at once, so we can search over the whole class.

## How many examples?

For many reasonable classes (e.g. linear approximators) the VC-dim is linear in the number of free parameters. Consequence: to learn well we need a number of examples **linear in the VC dimension**, so linear in the number of parameters in that case. This is where the usual rules of thumb of the form "at least $k$ examples per parameter" come from.

## [[Structural Risk Minimization]]

Choose among nested hypothesis spaces $H_1\subseteq H_2\subseteq\dots$ by minimizing the bound rather than the training error.

## Other analytical criteria

SRM is one **inductive principle** for complexity control; the others differ in what they use as the penalty, but all have the shape *fit term + complexity term*.

- **AIC / BIC** (Akaike / Bayesian Information Criterion): penalized likelihood, $-2\log\hat L+2k$ and $-2\log\hat L+k\ln N$ respectively, with $k$ the number of parameters. Cheap to compute, but limited to models that are **linear in the parameters**, and they need a likelihood. BIC penalizes complexity harder as $N$ grows, so it tends to pick smaller models than AIC.
- **MDL** (Minimum Description Length): prefer the hypothesis that minimizes the number of bits needed to describe the model **plus** the data given the model. A perfectly fitting but huge model costs as much as a small model with many residuals to encode.
- **Bayesian inference**: the prior over hypotheses plays the role of the complexity term.
- **[[Regularization]]**: Tikhonov, the practical version - the penalty is on $\lVert \boldsymbol{w}\rVert$ rather than on a counted capacity.

> [!note]
> The interesting comparison between them is *what they measure as complexity*: number of parameters (AIC/BIC), description length (MDL), VC-dim (SRM), norm of the weights (Tikhonov). They can disagree, because the number of parameters and the VC-dim are **not** the same thing - see [[VC Dimension]].

## Use and limits of the bound

Good for:

- a fundamental theoretical ground for principled ML, independent of model details or learning algorithm;
- highlighting the role of complexity control, and making it an inductive principle instead of trial and error;
- model selection/comparison, where only the **relative** size/ranking of the error matters;
- driving the development of new models (this is where [[SVM]] comes from).

Bad for:

- estimating the predictive error in practice - rarely used. The bounds are **worst case** (they must hold for all but a fraction $\delta$ of the possible functions and training sets) and can be very loose: the VC-confidence can be hundreds of times bigger than the actual overfitting effect, which makes it unsuitable for model assessment;
- classes whose VC-dim is hard to compute, typically anything with non-linear parameters;
- describing what very over-parametrized models actually do: past the interpolation threshold the test error can fall **again**, which the plain reading of the bound does not predict $\to$ **double descent**, see [[Deep Learning]].

## Related

- [[VC Dimension]]
- [[Regularization]] - Tikhonov as a practical SRM
- [[Bias-Variance Tradeoff]] - the same trade-off, other language
- [[Model Selection and Assessment]]
- [[Generalization and Risk]] - the risk and ERM this bound is about
