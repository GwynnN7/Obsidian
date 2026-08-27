---
tags: [machine-learning, ensembles]
---

# Ensemble Learning

Take advantage of several models instead of one: inputs go to $h_1,h_2,\dots,h_K$, and a **combiner** produces the final output.

## Voting schema (simplest)

- **Regression** - simple average committee:

  $$
  o(x)=\frac1K\sum_{i=1}^{K}h_i(x).
  $$

  The squared error of the committee is $\le$ the mean of the squared errors of the members:

  $$
  \operatorname{loss}(\mathbb{E}_i[h_i]) \le \mathbb{E}_i[\operatorname{loss}(h_i)].
  $$

  E.g. $t=5$, two models giving $2$ and $4$: mean error $((5-2)^2+(5-4)^2)/2=5$ vs committee $(5-3)^2=4$. Or $6$ and $4$: $((5-6)^2+(5-4)^2)/2=1$ vs $(5-5)^2=0$.

- **Classification** - vote over many classifiers, or classify after averaging the continuous outputs.

## Stacking schema

The combiner is itself a ML model, trained on the outputs of the members.

## When does it help

Only if the members are **diversified** - if they make the same errors there is no advantage. Two ways to differentiate them:

- [[Bagging]] - different bootstrap samples of the data.
- [[Boosting]] - concentrate each new learner on the previous errors.

Also useful for NN: the different runs with different random initializations are already an ensemble - see [[Multiple Minima]].

Related: [[Bias-Variance Tradeoff]].
