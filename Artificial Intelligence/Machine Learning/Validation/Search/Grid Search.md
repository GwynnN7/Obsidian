---
tags: [machine-learning, validation]
---

# Grid Search

Exhaustively generate the candidates from a grid of hyperparameter values, and evaluate each cell by model selection (validation set or inner CV).

Trivial example:

| | $\lambda=0.1$ | $\lambda=0.01$ | $\lambda=0.001$ |
|---|---|---|---|
| **1 unit** | Res1 | Res4 | Res7 |
| **10 units** | Res2 | Res5 | Res8 |
| **100 units** | Res3 | Res6 | Res9 |

If Res3 is the best, the winner is (units=100, $\lambda$=0.1).

- **Automatize it**, and parallelize: the trials are independent.
- With CV: run a k-fold CV **for each cell** of the grid, and compare the mean error over the validation sets of the folds.

## Cost

Cartesian product of the value sets:

$$
\#(\text{values in the range})^{\#\text{hyperparameters}} .
$$

$3\times3=9$ is fine; 5-6 hyperparameters with 10 values each is not.

## Two (or more) levels of nested grid search - useful

1. A first **coarse** grid search over combinations of exponentially growing values, to find good intervals/regions.
2. Then **finer** grid searches over smaller and smaller intervals - still using all the significant hyperparameters together.

## Why bother, if random search is cheaper

Grid search is very useful to **observe the behaviour** of the model as the hyperparameters change, i.e. to explore the hyperparameter space. That understanding is the point.

But it is computationally demanding for large spaces and does not scale with the number of hyperparameters $\to$ [[Random Search]].

See [[Hyperparameter Search]].
