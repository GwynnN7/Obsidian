# Momentum and Accelerated Gradient

Deflected gradient methods mix the current gradient with previous motion to reduce zig-zagging and exploit persistent directions.

## Nonlinear conjugate gradient

Use $d_k=-g_k+\beta_kd_{k-1}$. Common choices are
$$
\beta_k^{FR}=\frac{g_k^Tg_k}{g_{k-1}^Tg_{k-1}},\qquad
\beta_k^{PR}=\frac{g_k^T(g_k-g_{k-1})}{g_{k-1}^Tg_{k-1}}.
$$
On quadratics with exact line search these connect to linear CG; for general functions, line search and periodic restart are needed for robustness.

## Heavy ball

$$
x_{k+1}=x_k-\alpha\nabla f(x_k)+\beta(x_k-x_{k-1}),\qquad 0\le\beta<1.
$$
Momentum damps oscillation across narrow valleys and accelerates motion along consistent slopes. Poor $(\alpha,\beta)$ choices cause overshoot or divergence.

## Nesterov accelerated gradient

One common form evaluates at a look-ahead point:
$$
y_k=x_k+\beta_k(x_k-x_{k-1}),\qquad
x_{k+1}=y_k-\frac1L\nabla f(y_k).
$$
For convex $L$-smooth functions it achieves $f(x_k)-f^*=O(1/k^2)$, improving gradient descent's $O(1/k)$. For strongly convex problems, tuned acceleration improves condition-number dependence from roughly $\kappa$ to $\sqrt\kappa$.

Acceleration is not monotone: the objective may rise temporarily. Restarts help when oscillation is detected or strong-convexity parameters are unknown. Convex guarantees do not automatically transfer to noisy or nonconvex settings.

See [[Line Search and Fixed Steps]] and [[Gradient and Conjugate Gradient for Quadratics|Gradient and Conjugate Gradient for Quadratics]].


