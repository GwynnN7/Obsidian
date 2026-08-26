---
tags: [machine-learning, models, neural-networks]
---

# Activation Functions

$o_t=f_t(net_t)$. The choice decides what the unit (and the network) can do.

## The three basic ones

1. **Identity / linear** - $f(x)=x$. Derivative $=1$. Used for **regression output units**.
2. **Threshold (step)** - the LTU / [[Perceptron]]. Its derivative is not defined, which is exactly why it is not used with LMS.
3. **Sigmoidal (squashing)** - smooth, differentiable approximation of the threshold.

## Sigmoids

**Logistic**, range $[0,1]$:

$$
f_\sigma(x)=\frac{1}{1+e^{-ax}},\qquad \frac{df}{dx}=f(x)(1-f(x))\ \ (a=1).
$$

**TanH** (symmetric), range $[-1,+1]$:

$$
f_{symm}(x)=2f_\sigma(x)-1=\tanh(ax/2),\qquad \frac{df}{dx}=1-f_{\tanh}(x)^2\ \ (a=1).
$$

- $a$ is the slope. $a\to0$ gives a linear unit, $a\to\infty$ gives an LTU.
- They are semi-linear close to $0$.
- Often tanh learns faster (Haykin).

### Using them as classifiers

- Logistic: output $\ge0.5$ $\to$ positive class, $<0.5$ $\to$ negative. TanH: threshold at $0$.
- The threshold can be moved (study the effect on FP/FN, or use a [[ROC Curve]]).
- A **rejection zone** around the threshold is possible, to avoid fragile decisions.

### Effect of the derivative on training

$\delta_p=(d_p-o(x_p))f'_\sigma$, so:

- $f'_\sigma$ is **maximum** for $net\approx0$ (the near-linear zone): high $\delta$, big steps;
- $f'_\sigma$ is **minimum** in the saturated zones: tiny $\delta$, very slow weight changes.

Hence: avoid premature saturation - start with **small weights** (see [[Weight Initialization]]), and keep the target inside the output range (use $0.9$ instead of $1$, $-0.9$ instead of $-1$, to avoid asymptotic convergence).

## Others

- **RBF** - $f(x)=e^{-ax^2}$, gaussian, on $\lVert w-x_{input}\rVert$ $\to$ RBF networks.
- **Softmax** - multi-output, sums to 1, readable as a probability. See [[Multiclass Classification]].
- **ReLU** - $f(x)=\max(0,x)$. The default choice for deep models, discussed in [[Deep Learning]].
- **Softplus** - $\ln(1+e^x)$, smooth approximation of ReLU.
- **Tanh-like piecewise linear** - for efficient computation.
- **Stochastic neurons** - output $+1$ with probability $P(net)$ $\to$ Boltzmann machines.

Many of them perform comparably.

## Choosing the output units

- **Regression:** linear unit(s), one per quantitative response.
- **Classification:** 1 unit (binary) or 1-of-K with multi-output; sigmoid + threshold, or softmax + cross-entropy.

> [!important]
> Non-linear hidden units are **essential**: an MLP with linear units collapses into a single-layer network (exercise: rewrite $h(x)$ with $f=$ identity).

Related: [[Neural Networks]], [[Backpropagation]].
