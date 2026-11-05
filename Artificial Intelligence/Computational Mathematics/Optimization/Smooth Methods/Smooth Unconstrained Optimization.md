# Smooth Unconstrained Optimization

For differentiable $f:\mathbb R^n\to\mathbb R$, most methods use
$$x_{k+1}=x_k+\alpha_kd_k.$$
A direction is a descent direction when $\nabla f(x_k)^Td_k<0$; then sufficiently small positive steps decrease $f$.

## Gradient method

Choose $d_k=-g_k$, $g_k=\nabla f(x_k)$:
$$x_{k+1}=x_k-\alpha_kg_k.$$
It is cheap and globally useful, but sensitive to scaling and conditioning. Exact line search minimizes $f(x_k-\alpha g_k)$, but may cost too much. Practical methods use an inexact search or a justified fixed step.

## Scaled directions

A broader family uses $d_k=-B_kg_k$, where $B_k\succ0$. This remains descent because $g_k^Td_k=-g_k^TB_kg_k<0$. Choices include:

- $B_k=I$: gradient descent;
- $B_k=\nabla^2f(x_k)^{-1}$: Newton (when positive definite);
- an inverse-Hessian approximation: quasi-Newton;
- a preconditioner: cheap scaling correction.

## Global convergence versus rate

Global convergence usually means every accumulation point is stationary or $\|g_k\|\to0$, not that a nonconvex global minimum is found. Typical local rates are sublinear (convex gradient), linear (strongly convex gradient), superlinear (quasi-Newton), and quadratic (Newton).

## Stopping criteria

Combine scaled gradient norm, step norm, objective change, and iteration/evaluation limits. Report why the solver stopped. A small step can mean convergence, bad scale, or failed line search.

Continue with [[Line Search and Fixed Steps]], [[Newton and Quasi-Newton Methods]], and [[Momentum and Accelerated Gradient]].


