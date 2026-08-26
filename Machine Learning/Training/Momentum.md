---
tags: [machine-learning, training, hyperparameter]
---

# Momentum

Add a fraction of the previous update to the current gradient step ("heavy ball method"):

$$
\Delta w_{new}=-\eta\frac{\partial E(w)}{\partial w}+\alpha\,\Delta w_{old},
\qquad w_{new}=w+\Delta w_{new},
$$

then save $\Delta w_{old}\leftarrow\Delta w_{new}$ for the next step. Note $\eta$ is **inside** $\Delta w$ here.

$0<\alpha<1$, typically $[0.5,\ 0.9]$.

## Effects

- **Faster on plateaus:** same sign of the gradient over consecutive iterations $\to$ the step grows.
- **Damps oscillations:** stabilizing in the directions whose sign keeps flipping.
- **Inertia:** allows a higher $\eta$.
- Picture: in a canyon of the error surface (poor conditioning of the Hessian) the momentum path travels lengthwise towards the minimum instead of zig-zagging between the walls.

## Batch vs on-line

- Born with batch learning, and commonly assumed to help more there.
- **On-line:** $\Delta w_{old}$ is the $\Delta w_{p-1}$ of the previous **example**, so it becomes a moving average of the past gradients that smooths the stochastic samples inside the epoch, e.g. $grad=\alpha\,grad+(1-\alpha)grad_{now}$. A different meaning from the batch version.
- **Mini-batch:** again a moving average over successive mini-batches (different examples each time).

Like any heuristic: check whether it is really needed.

## Nesterov momentum

Evaluate the gradient **after** the momentum is applied:

1. apply the momentum: $w'=w+\alpha\Delta w_{old}$;
2. evaluate the new gradient at this interim point $w'$;
3. compute and apply $\Delta w$ to the original $w$ (momentum + new gradient).

Improves the rate of convergence for **batch** mode (not for the stochastic case). Inspired by Nesterov's accelerated gradient, with nice formal properties in the convex case.

Related: [[Gradient Descent]], [[Learning Rate]], [[Regularization]].
