---
tags: [machine-learning, validation]
---

# Regression Metrics

Given the residual $r_i=(y_i-o_i)$ with $y$ the target and $o$ the model output:

- **[[Mean Squared Error]]** - $\operatorname{mean}_i[r_i^2]$, and its root **RMSE** (or $S$).
- **Mean absolute error** - $\operatorname{mean}_i[|r_i|]$.
- **Max absolute error** - $\max_i|r_i|$.
- **Correlation coefficient $R$** (and $R^2$, the coefficient of determination):

  $$
  R=\sqrt{1-\frac{S^2}{S_y^2}},\qquad S_y^2=\operatorname{mean}_i\big[(y_i-\operatorname{mean}_i[y])^2\big]\ \ (\text{variance of } y).
  $$

  It is the degree of linear dependence between $y$ and $o$, in $[0,1]$, where 1 is the best.

Also useful:

- **outputs vs targets plots**,
- various statistical tests and significance analysis.

Related: [[Regression]], [[Classification Metrics]], [[Validation]].
