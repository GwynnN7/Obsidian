---
tags: [machine-learning, models, algorithms]
---

# Cascade Correlation

The number of hidden units is a [[Model Selection and Assessment|model selection]] issue. Two ways to let the algorithm decide it instead:

- **Constructive / incremental:** start minimal, add units and connections during training (Tower, Tiling, Upstart for classification; Cascade Correlation for both regression and classification).
- **Pruning:** start with a large network and progressively remove weights or units.

## The result

CC learns the weights **and** the topology, by interleaving two objectives and training one unit at a time:

| | what is optimized | over which weights |
|---|---|---|
| **1. output step** | *minimize* the total error (LMS, by [[Backpropagation]]) | the output-layer weights only |
| **2. candidate step** | *maximize* the correlation $S$ of a new unit with the residual error | the new unit's incoming weights |

$$
S=\sum_k\Big|\sum_p\big(o_p-\operatorname{mean}_p(o_p)\big)\big(E_{p,k}-\operatorname{mean}_p(E_{p,k})\big)\Big|,
\qquad E_{p,k}=o_{p,k}-d_{p,k}
$$

with $p$ the pattern, $k$ the output unit, $o$ the candidate's output and $E$ the residual error at the output layer.

> [!important] Freezing is the whole trick
> Once a candidate is installed its incoming weights are **frozen forever**. So there is never a gradient propagated through the hidden layers - each step is a one-unit training problem, and the network grows one permanent feature detector at a time.

## The CC algorithm

1. Start with $N_0$, a network **without hidden units**. Train it, compute the error. If it cannot solve the problem, go to $N_1$.
2. In $N_1$ add a hidden unit whose **correlation (covariance) with the residual error** of $N_0$ is maximized, by training its weights.
3. After training, the weights of the new unit are **frozen**; only the remaining (output-layer) weights are retrained.
4. If $N_1$ is still not enough, add more hidden units, each connected to all the inputs **and to all the previously installed hidden units** (hence the "cascade").
5. Stop when the residual error meets the criterion.

So CC learns both the weights **and** the topology: it deals with a hypothesis space of flexible size, and trains a single unit at each step.

## Maximizing the correlation

Differentiating $S$ with respect to the candidate's incoming weights gives:

$$
\frac{\partial S}{\partial w_j}=\sum_k \operatorname{sgn}(S_k)\sum_p (E_{p,k}-\operatorname{mean}_p(E_{p,k}))\,f'(net_{p,h})\,I_{p,j}
$$

with $h$ the candidate index and $I$ its input. The algorithm interleaves minimizing the total error (LMS on the output layer, by [[Backpropagation]]) and maximizing $S$ for the new candidate.

## Notes

- Each hidden unit reduces the residual output error: it solves a specific sub-problem and becomes a permanent **feature detector**.
- **POOL:** the maximization is a gradient ascent on a surface with several maxima, so a pool of candidate units is trained and the best one is kept - this helps avoid local maxima.
- It is a **greedy** algorithm: easy convergence, possibly a minimal number of units, but it can lead into [[Overfitting]] $\to$ needs [[Regularization]].
- The loss need not be "max $S$": for regression, LMS works too.

Related: [[Neural Networks]], [[Hyperparameter Guidelines]]. For the evolutionary way of growing an architecture, see [[NEAT]].
