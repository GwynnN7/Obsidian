# Optimization

Optimization studies
$$
\min_{x\in X} f(x),
$$
where $f$ is the objective and $X$ is the feasible set. This part of the course is **unconstrained** after the introductory definitions; constrained, nonsmooth, and duality chapters from the larger official text are outside the taught scope.

## Course map

### Foundations

- [[Optimization Problems|Optimization Problems]] - solutions, values, feasibility, reformulations, tolerances.
- [[Elementary Optimization Models|Elementary Optimization Models]] - linear and quadratic examples.
- [[Quadratic Optimization|Quadratic Optimization]] - definiteness and optimality of quadratics.
- [[Gradient and Conjugate Gradient for Quadratics|Gradient and Conjugate Gradient for Quadratics]].
- [[Cholesky and Polynomial Interpolation|Cholesky and Polynomial Interpolation]].

### Univariate optimization

- [[Univariate Optimization|Univariate Optimization]] - local/global minima and bracketing.
- [[Bracketing and Dichotomic Search|Bracketing and Dichotomic Search]].
- [[Interpolation Newton and Convexity|Interpolation Newton and Convexity]].

### Multivariate optimality

- [[Multivariate Derivatives|Multivariate Derivatives]].
- [[Taylor Models and Optimality Conditions|Taylor Models and Optimality Conditions]].
- [[Convex Functions|Convex Functions]].
- [[Neural Networks as Optimization Models|Neural Networks as Optimization Models]].

### Smooth algorithms

- [[Smooth Unconstrained Optimization|Smooth Unconstrained Optimization]].
- [[Line Search and Fixed Steps|Line Search and Fixed Steps]].
- [[Newton and Quasi-Newton Methods|Newton and Quasi-Newton Methods]].
- [[Momentum and Accelerated Gradient|Momentum and Accelerated Gradient]].

## Core pattern

Most methods generate $x_{k+1}=x_k+\alpha_kd_k$. Their differences are:

- how the direction $d_k$ is chosen;
- how the step $\alpha_k$ is chosen;
- what assumptions guarantee descent and convergence;
- how much one iteration costs.


