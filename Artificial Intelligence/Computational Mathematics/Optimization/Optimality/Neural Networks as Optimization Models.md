# Neural Networks as Optimization Models

A feed-forward network composes affine maps and nonlinear activations:
$$
h_0=x,\qquad h_{\ell}=\sigma_{\ell}(W_{\ell}h_{\ell-1}+b_{\ell}),\qquad \hat y=h_L.
$$
Training minimizes an empirical loss over parameters $\theta=(W_\ell,b_\ell)$:
$$
F(\theta)=\frac1N\sum_{i=1}^N\ell(\hat y(x_i;\theta),y_i)+R(\theta).
$$

## Why it belongs here

- The chain rule/Jacobians in [[Multivariate Derivatives]] produce backpropagation.
- The problem is high-dimensional, smooth for smooth activations, and generally nonconvex.
- Symmetries such as permuting hidden units create equivalent solutions.
- Hessians can be indefinite and poorly conditioned, so stationarity need not mean a minimum.

## Backpropagation viewpoint

Backpropagation is reverse-mode differentiation: it computes vector-Jacobian products from the loss backward, reusing intermediate activations. Its cost is comparable to a small constant number of forward evaluations, far below forming a full Jacobian.

Full gradient descent uses all samples per step; stochastic/mini-batch variants use cheap noisy estimates. The course's smooth methods explain descent directions, stepsizes, momentum, and curvature approximations, though large-network training adds stochastic issues beyond this module.

See [[Smooth Unconstrained Optimization|Smooth Unconstrained Optimization]] and [[Momentum and Accelerated Gradient|Momentum and Accelerated Gradient]].


