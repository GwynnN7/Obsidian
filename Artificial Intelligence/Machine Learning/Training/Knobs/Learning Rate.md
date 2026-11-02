---
tags: [machine-learning, training]
---

# Learning Rate

$\eta$, the step size of [[Gradient Descent]].

- **High $\eta$:** fast, but possibly unstable if too large (the [[Learning Curve]] oscillates or diverges).
- **Low $\eta$:** stable, but possibly far too slow (curve stuck).
- Typical search range in LMS: $[0.01,\ 0.5]$ - but always by [[Grid Search]], not by faith.

## Interaction with batch size

- **Batch:** more accurate gradient $\to$ higher $\eta$.
- **On-line:** noisier $\to$ **smaller** $\eta$.

## LMS: divide by $l$

Use the **mean** of the gradients over the epoch (sum of gradients $/\,l$): it makes $\eta$ independent of the number of input data. This is what "Least Mean Squares" means, and it is equivalent to using $\eta/l$.

> [!warning]
> If you compare on-line with batch-using-LMS, the on-line version needs a **much** smaller $\eta$ - by roughly the same factor $1/l$ used to divide the batch gradient. Otherwise the comparison is meaningless.
> For mini-batch, either divide by $mb$ and retune $\eta$, or simply divide $\eta$ by $l$ as well.

With LMS, $0<\eta<1$.

## Improvements

1. **[[Momentum]]** (and Nesterov) - the first thing to add.
2. **Variable learning rate.** With mini-batch the gradient does not go to zero near a minimum (unlike the exact gradient), so a fixed $\eta$ should be avoided. Decay linearly until iteration $\tau$, with $s$ the step:

   $$
   \eta_s=\Big(1-\frac{s}{\tau}\Big)\eta_0+\frac{s}{\tau}\eta_\tau ,
   $$

   then keep $\eta$ fixed and small ($=\eta_\tau$). Set-up: $\eta_\tau\approx1\%$ of $\eta_0$, $\tau$ a few hundred steps. And $\eta_0$? Again the no-instability / no-stuck trade-off, found by preliminary trials or grid search.
3. **Adaptive learning rates** - adapt $\eta$ during training, one per parameter: AdaGrad, RMSProp, Adam, AdamW. See [[Optimizers]] for what each one actually does.
4. **Per-layer** in deep networks: higher for deep layers, or based on fan-in (higher for few input connections).

Related: [[Batch Online and Mini-Batch]], [[Hyperparameter Guidelines]].
