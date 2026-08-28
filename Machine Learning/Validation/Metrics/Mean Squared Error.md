---
tags: [machine-learning, validation]
---

# Mean Squared Error

Average squared prediction error over a set of $r$ patterns:

$$
\operatorname{MSE}=\frac1r\sum_{p=1}^{r}(y_p-h(x_p))^2 .
$$

- The inner loss is the squared error - see [[Loss Functions]].
- **RMSE** is its square root, back on the scale of the target.
- Dividing by $r$ (rather than just summing) is what makes it "mean", and it also makes the gradient - and hence $\eta$ - independent of the number of data. See [[Learning Rate]].

Common in [[Regression]], [[Linear Models]] and neural-network training. Careful with the [[Loss Functions|Loss vs Error]] distinction once a penalty term is added.

See [[Regression Metrics]].
