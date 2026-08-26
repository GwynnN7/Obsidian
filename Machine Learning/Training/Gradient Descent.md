---
tags: [machine-learning, training]
---

# Gradient Descent

Local search: iteratively move the parameters in the negative-gradient direction to reduce a differentiable objective.

$$
\Delta \boldsymbol{w}=-\frac{\partial E(\boldsymbol{w})}{\partial \boldsymbol{w}},\qquad \boldsymbol{w}_{new}=\boldsymbol{w}+\eta\,\Delta \boldsymbol{w} ,
$$

or component-wise on each $w_j$. $\eta$ is the step size ([[Learning Rate]]).

- The gradient points in the direction of **steepest ascent**; its magnitude gives the steepness. We move against it.
- Local minimum = stationary point, gradient zero.
- For [[Linear Models]] the error surface is a convex paraboloid $\to$ no local minima. For [[Neural Networks]] it is not - see [[Multiple Minima]].

## Why it is the important algorithm

- It searches an infinite (continuous) hypothesis space.
- It applies to **any** continuous $H$ with a differentiable loss - not just linear models. This is what makes [[Backpropagation]] and [[Deep Learning]] possible.
- It is cheap: linear in the number of free parameters.

## Faster alternatives

- **Second order** (Hessian): Newton's methods, Gauss-Newton, Levenberg-Marquardt. More information about the curvature, but the Hessian can be far too large.
- **Hessian-free:** conjugate gradients (can be nice for faster convergence), quasi-Newton / BFGS.
- **NN-specific heuristics:** Quick-prop (jump to the minimum of a locally estimated parabola), R-prop (uses only the **sign** of the gradient, so it does not vanish in deep layers).
- **Gradient clipping** bounds the step without changing its direction, for the cliffs that appear with many layers $\to$ [[Deep Learning]].

> [!note] Saddle points
> In high-dimensional spaces saddle points (gradient $\approx0$ but not a minimum) can grow exponentially w.r.t. local minima. Newton-type methods search for zero-gradient points and so are **more** likely to get stuck there - which explains why standard second-order methods never replaced gradient descent for NN. Gradient descent empirically escapes saddle points.

In practice: SGD with [[Momentum]] plus [[Regularization]] is the starting point, and the optimization technique only matters when there are actual obstacles to training.

Related: [[Training]], [[Batch Online and Mini-Batch]].
