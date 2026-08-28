---
tags: [machine-learning, training, algorithms]
---

# Optimizers

All of these are variants of [[Gradient Descent]]: they differ in *how* the raw gradient $g=\partial \text{Loss}/\partial w$ is turned into a step.

## The family

Every one of them computes the same $g$ and differs only in the step it builds from it.

| | Step $\Delta w$ | Idea | State | Hyperparameters |
|---|---|---|---|---|
| **SGD** | $-\eta g$ | follow the gradient | none | $\eta$ |
| **SGD + [[Momentum]]** | $\alpha\,\Delta w_{old}-\eta g$ | inertia from past steps | 1 (velocity) | $\eta$, $\alpha\in[0.5,0.9]$ |
| **Nesterov** | as above, but $g$ measured **after** the momentum step | look ahead | 1 | $\eta$, $\alpha$ |
| **AdaGrad** | $-\dfrac{\eta}{\sqrt{r}+\epsilon}g$, &nbsp;$r\leftarrow r+g^2$ | per-weight $\eta$ from accumulated $g^2$ | 1 | $\eta$ |
| **RMSProp** | same step, &nbsp;$r\leftarrow\rho r+(1-\rho)g^2$ | *forget* old gradients | 1 | $\eta$, $\rho\approx0.9$ |
| **Adam** | $-\eta\dfrac{\hat m}{\sqrt{\hat r}+\epsilon}$ | RMSProp + momentum + bias correction | 2 | $\eta$, $\beta_1$, $\beta_2$ |
| **AdamW** | Adam step $-\ \eta\lambda w$ | Adam with **decoupled** weight decay | 2 | $\eta$, $\beta_1$, $\beta_2$, $\lambda$ |

Read down the first column and the whole family is one line each; the sections below say where each piece comes from.

## AdaGrad

Accumulate the squared gradients and scale each weight's step by the inverse root:

$$
r \leftarrow r + g^2, \qquad \Delta w=-\frac{\eta}{\sqrt{r}+\epsilon}\,g .
$$

- Gives a **per-parameter** learning rate: rarely-updated weights get bigger steps.
- **Problem:** $r$ only grows, so the effective step decays monotonically and training can stall before convergence.

## RMSProp

Fixes AdaGrad by replacing the sum with an exponentially weighted **moving average**:

$$
r \leftarrow \rho\,r+(1-\rho)\,g^2, \qquad \Delta w=-\frac{\eta}{\sqrt{r}+\epsilon}\,g
$$

with $\rho\approx0.9$. Old gradients are forgotten, so the step size does not die out. Popular for deep and recurrent models.

## Adam

*Adaptive Moment Estimation* - RMSProp plus momentum, i.e. a moving average of both the first moment (the gradient) and the second (its square):

$$
m \leftarrow \beta_1 m+(1-\beta_1)g, \qquad r \leftarrow \beta_2 r+(1-\beta_2)g^2
$$

Both start at zero, so they are biased towards zero at the beginning; hence the **bias correction**:

$$
\hat m=\frac{m}{1-\beta_1^{\,t}},\qquad \hat r=\frac{r}{1-\beta_2^{\,t}},
\qquad
\Delta w=-\eta\,\frac{\hat m}{\sqrt{\hat r}+\epsilon}
$$

Usual defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\eta=10^{-3}$. Often robust out of the box, which is exactly why it is worth being suspicious of it - see the warning below.

## AdamW

The problem: with Adam, adding an L2 penalty to the **gradient** does *not* give true weight decay, because that penalty term is then divided by $\sqrt{\hat r}$ like everything else. Weights with large gradient history get decayed **less**, which is the opposite of what you want, and $\lambda$ stops being independent of $\eta$.

AdamW *decouples* them: the adaptive machinery sees only the data gradient, and the decay is applied straight to the weights:

$$
w \leftarrow w-\eta\,\frac{\hat m}{\sqrt{\hat r}+\epsilon}-\eta\lambda w
$$

This is the same principle already used in [[Regularization]]: keep $\eta$, $\alpha$ and $\lambda$ independent so their effects can be controlled separately. Use AdamW whenever you would have used Adam + L2.

## Which one

> [!warning]
> An adaptive optimizer reduces the fine-tuning of $\eta$, it does not remove the need for [[Model Selection and Assessment|model selection]] - and it does **nothing** for complexity control. The generalization work is still done by [[Regularization]], [[Early Stopping]] and the number of units.

- Start with **SGD + momentum**: it is the didactic baseline, and it often generalizes at least as well.
- Move to **RMSProp / AdamW** when the plain version struggles (deep nets, recurrent nets, very noisy gradients).
- Whatever you pick, still look at the [[Learning Curve]] and still tune $\eta$ - the adaptive methods have their own $\eta$ and it matters.
- The optimizer only matters when there are real obstacles to training; validation matters more.

Related: [[Learning Rate]], [[Batch Online and Mini-Batch]], [[Training]], [[Hyperparameter Guidelines]].
