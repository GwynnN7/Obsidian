---
tags: [machine-learning, models]
---

# Activation Functions

$o_t=f_t(net_t)$ - the function a unit applies to its weighted input. The choice decides what the unit, and therefore the network, can do.

## The table

| Name | $f(x)$ | $f'(x)$ | Range | Where it is used |
|---|---|---|---|---|
| **Identity / linear** | $x$ | $1$ | $\mathbb{R}$ | regression output units |
| **Threshold / step** (LTU) | $1$ if $x>0$, else $0$ (or $\operatorname{sign}(x)$) | undefined at $0$, $0$ elsewhere | $\{0,1\}$ or $\{-1,+1\}$ | [[Perceptron]]; **not** usable with gradient methods |
| **Logistic / sigmoid** | $\dfrac{1}{1+e^{-ax}}$ | $f(x)\,(1-f(x))$ &nbsp;$(a=1)$ | $(0,1)$ | classic hidden units, binary output |
| **TanH** | $\tanh(x)=\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ | $1-f(x)^2$ | $(-1,+1)$ | classic hidden units; usually learns faster than logistic |
| **ReLU** | $\max(0,x)$ | $1$ if $x>0$, $0$ if $x<0$, undefined at $0$ | $[0,+\infty)$ | **the default for deep models** |
| **Leaky ReLU** | $x$ if $x>0$, else $\alpha x$ &nbsp;($\alpha\approx0.01$) | $1$ if $x>0$, else $\alpha$ | $\mathbb{R}$ | fixes dead units |
| **ELU** | $x$ if $x>0$, else $\alpha(e^{x}-1)$ | $1$ if $x>0$, else $f(x)+\alpha=\alpha e^{x}$ | $(-\alpha,+\infty)$ | smooth version of the same fix |
| **Softplus** | $\ln(1+e^{x})$ | $\dfrac{1}{1+e^{-x}}$ (the logistic) | $(0,+\infty)$ | smooth approximation of ReLU |
| **Gaussian / RBF** | $e^{-ax^2}$, applied to $\lVert \boldsymbol{w}-\boldsymbol{x}\rVert$ | $-2ax\,e^{-ax^2}$ | $(0,1]$ | RBF networks, local units |

**Softmax** is the odd one out - it maps a whole vector to a whole vector, so it has no single-argument formula:

$$
o_k(\boldsymbol{x})=\frac{e^{net_k}}{\sum_{j=1}^{K}e^{net_j}},
\qquad
\frac{\partial o_k}{\partial net_j}=o_k(\delta_{kj}-o_j)
$$

The outputs are positive and sum to $1$, so they read as $p(\text{class}=k\mid \boldsymbol{x})$. Output layer for multi-class problems, paired with cross-entropy - see [[Multiclass Classification]].

> [!important] Non-linear hidden units are essential
> An MLP with linear hidden units collapses into a single-layer network: a composition of linear maps is linear. (Exercise: rewrite $h(\boldsymbol{x})$ with $f=$ identity.) Everything the depth buys depends on $f$ being non-linear.

## The three basic families

1. **Identity / linear** - no squashing at all.
2. **Threshold (step)** - the LTU of the [[Perceptron]]. Its derivative is not defined, which is exactly why it cannot be used with LMS or [[Backpropagation]].
3. **Sigmoidal (squashing)** - a smooth, differentiable approximation of the threshold. This is the family that made gradient training of networks possible.

## Sigmoids in detail

The two are the same function rescaled:

$$
f_{symm}(x)=2f_\sigma(x)-1=\tanh(ax/2)
$$

- $a$ is the **slope**. $a\to0$ gives a linear unit, $a\to\infty$ gives an LTU - the sigmoid interpolates between the two.
- Both are semi-linear close to $0$.
- TanH is symmetric around zero, which is usually why it trains faster.

### Using them as classifiers

- Logistic: output $\ge0.5$ $\to$ positive class, $<0.5$ $\to$ negative. TanH: threshold at $0$.
- The threshold can be moved - study the effect on FP/FN, or use a [[ROC Curve]].
- A **rejection zone** around the threshold is possible, to avoid fragile decisions.

### Why the derivative matters: saturation

In [[Backpropagation]] the delta of an output unit is $\delta_p=(d_p-o(\boldsymbol{x}_p))\,f'_\sigma$, so the derivative gates every weight update:

- $f'_\sigma$ is **maximum** for $net\approx0$, the near-linear zone: large $\delta$, big steps;
- $f'_\sigma$ is **minimum** in the **saturated** zones: tiny $\delta$, and learning nearly stops.

Two practical consequences:

- start with **small weights** so units begin in the linear zone - see [[Weight Initialization]];
- keep the targets inside the output range (use $0.9$ instead of $1$, $-0.9$ instead of $-1$), otherwise the unit chases an asymptote it can never reach.

Stacked over many layers, the same shrinking factor is what produces the **vanishing gradient** - see [[Deep Learning]].

## ReLU and its variants

ReLU is the default in deep models because it fixes exactly the problem above:

- the derivative is **exactly $1$** on the positive side, so nothing shrinks the gradient as it propagates back;
- it is cheap (a comparison), and it gives **sparse activation** - roughly half the units are off for a given input;
- not differentiable at $0$: libraries just pick $0$ or $1$ there, and it never matters in practice.

The cost is the **dead unit**: a unit stuck with $net<0$ for every pattern has zero gradient forever and never recovers. Hence:

- **Leaky ReLU** - a small slope $\alpha$ on the negative side keeps a gradient alive;
- **ELU** - the same fix but smooth, and with negative outputs that push the mean activation toward zero;
- a common trick: initialize the biases to a small positive value (e.g. $0.1$) so units start active.

## Other units

- **Tanh-like piecewise linear** - a cheap approximation for efficient computation.
- **Stochastic neurons** - output $+1$ with probability $P(net)$ $\to$ Boltzmann machines.

## Choosing the output units

- **Regression:** linear unit(s), one per quantitative response.
- **Classification:** one unit for binary (sigmoid + threshold), or 1-of-K multi-output with **softmax + cross-entropy**.
- **Hidden units:** tanh historically, ReLU (or a variant) for anything deep.

Related: [[Neural Networks]], [[Backpropagation]], [[Deep Learning]], [[Weight Initialization]].
