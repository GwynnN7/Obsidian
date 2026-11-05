# Elementary Optimization Models

Simple examples reveal the cases every general method must handle.

## Univariate linear and quadratic functions

For $f(x)=cx$ on $\mathbb R$:

- if $c\ne0$, the problem is unbounded below;
- if $c=0$, every point is optimal.

On an interval $[\ell,u]$, the minimum is an endpoint: $\ell$ if $c>0$, $u$ if $c<0$.

For $f(x)=\frac12qx^2-bx+c$:

- $q>0$: unique minimizer $x^*=b/q$;
- $q=0$: linear/constant case;
- $q<0$: unbounded below on $\mathbb R$ (the stationary point is a maximum).

Completing the square gives
$$
f(x)=\frac q2\left(x-\frac bq\right)^2+c-\frac{b^2}{2q}.
$$

## Multivariate linear functions

$f(x)=c^Tx$ is unbounded below on $\mathbb R^n$ unless $c=0$. Over a box, each coordinate independently chooses the bound favored by the sign of $c_i$.

## Separable functions

If $f(x)=\sum_i f_i(x_i)$ and the feasible region is a Cartesian product, minimize each scalar component separately. Separability can turn one large problem into many independent small ones.

## Quadratic functions

The standard form is
$$
f(x)=\frac12x^TQx-b^Tx+c,
$$
where only the symmetric part of $Q$ matters because
$$x^TQx=x^T\frac{Q+Q^T}{2}x.$$
Its gradient is $Qx-b$ when $Q$ is symmetric. The solution behavior is governed by the eigenvalues of $Q$; see [[Quadratic Optimization]].

## Why these examples matter

- They separate existence, uniqueness, and boundedness.
- They show why a stationary point need not be a minimum.
- They introduce curvature: positive curvature stabilizes minimization; negative curvature gives descent to $-\infty$.
- Quadratic models are the local models used by [[Newton and Quasi-Newton Methods|Newton and Quasi-Newton Methods]].


