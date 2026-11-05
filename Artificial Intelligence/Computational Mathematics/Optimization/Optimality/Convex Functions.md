# Convex Functions

$f:\mathbb R^n\to\mathbb R$ is convex when
$$
f(\theta x+(1-\theta)y)\le\theta f(x)+(1-\theta)f(y),\quad \theta\in[0,1].
$$
Geometrically, its graph lies below every chord.

## Equivalent smooth tests

For differentiable $f$:
$$f(y)\ge f(x)+\nabla f(x)^T(y-x).$$
The tangent hyperplane is a global underestimator. For twice differentiable $f$ on a convex domain:
$$f\text{ convex}\iff \nabla^2f(x)\succeq0\quad\forall x.$$

$\mu$-strong convexity strengthens the lower bound:
$$
f(y)\ge f(x)+\nabla f(x)^T(y-x)+\frac\mu2\|y-x\|^2.
$$
Strong convexity implies a unique minimizer and supports linear convergence rates.

## Optimization consequences

- Every local minimum is global.
- Every stationary point is a global minimum.
- Strict convexity gives at most one minimizer.
- The minimizer set of a convex function is convex.

<details>
<summary>Why a local minimum is global</summary>

Suppose local minimizer $x^*$ is not global, so some $y$ has $f(y)<f(x^*)$. Points $x_\theta=(1-\theta)x^*+\theta y$ approach $x^*$ as $\theta\downarrow0$. Convexity gives $f(x_\theta)<f(x^*)$, contradicting local minimality.

</details>

Nonnegative weighted sums and pointwise maxima of convex functions are convex. Affine composition $f(Ax+b)$ preserves convexity. Norms are convex; squared Euclidean norm is strongly convex.

Convexity makes algorithms safer, but conditioning still controls speed. See [[Line Search and Fixed Steps|Line Search and Fixed Steps]].


