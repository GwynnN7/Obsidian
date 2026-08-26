---
tags: [machine-learning, theory]
---

# Overfitting

The learner outputs a hypothesis with very low training error that does not generalize.

Formally: $h\in H$ has true risk $R$ and empirical (training) error $E$, but there is another $h'\in H$ with

$$
E' > E, \qquad R' < R,
$$

so $h'$ fits the training data worse and is the better model.

## Why it happens

- $H$ too flexible for the amount of data: the model fits the noise too.
- In SLT terms: growing capacity lowers $R_{emp}$ but raises the VC-confidence, so the bound on $R$ goes up. See [[Statistical Learning Theory]].
- In bias-variance terms: low bias, high variance. See [[Bias-Variance Tradeoff]].

> [!note]
> Noise makes overfitting easier to see, but it is **not** the cause - the cause is the complexity trade-off. You can overfit perfectly clean data (e.g. a high-degree polynomial through few exact points).

## Controls

- [[Regularization]] - penalty on the loss, the main tool.
- [[Early Stopping]] - empirical, needs validation data.
- Reduced capacity: fewer units / lower degree / [[Feature Selection]].
- [[Model Selection and Assessment]] with validation data to find the trade-off.
- [[Bagging]] for high-variance models.

See also [[Underfitting]].
