# Interpolation Newton and Convexity

## Parabolic interpolation

Fit a quadratic through three sampled points and use its minimizer as the next trial point. It can be much faster than fixed-ratio interval reduction when $f$ is locally smooth and quadratic-like. It can fail when the fitted parabola is nearly linear, opens downward, or proposes a point outside the bracket, so safeguard it.

## Newton's method for minimization

Apply root finding to $f'(x)=0$:
$$
x_{k+1}=x_k-\frac{f'(x_k)}{f''(x_k)}.
$$
Equivalently, minimize the local quadratic Taylor model. Near a nondegenerate minimizer with $f''(x^*)>0$ and sufficiently smooth $f$, convergence is quadratic.

Failure modes include a zero/tiny $f''(x_k)$, negative curvature, a start outside the convergence basin, or a step outside the function domain. Use damping/line search or retain a bracket and fall back to a safe step.

<details>
<summary>Local quadratic convergence sketch</summary>

Taylor-expand $f'$ around $x^*$ where $f'(x^*)=0$ and $f''(x^*)\ne0$. With $e_k=x_k-x^*$,
$$f'(x_k)=f''(x^*)e_k+O(e_k^2),\qquad f''(x_k)=f''(x^*)+O(e_k).$$
Substitution into Newton's update cancels the linear term, giving $e_{k+1}=O(e_k^2)$.

</details>

## Convexity in one dimension

$f$ is convex if
$$f(\theta x+(1-\theta)y)\le\theta f(x)+(1-\theta)f(y).$$
For differentiable $f$, this is equivalent to $f(y)\ge f(x)+f'(x)(y-x)$. If twice differentiable, $f''(x)\ge0$ implies convexity. Every local minimum of a convex function is global; under strict convexity it is unique.

The multivariate version is in [[Convex Functions|Convex Functions]].


