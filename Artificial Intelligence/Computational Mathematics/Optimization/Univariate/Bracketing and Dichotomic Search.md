# Bracketing and Dichotomic Search

## Dichotomic interval reduction

For a unimodal function on $[a,b]$, pick
$$
x_1=\frac{a+b}{2}-\delta,\qquad x_2=\frac{a+b}{2}+\delta.
$$

- If $f(x_1)<f(x_2)$, discard $(x_2,b]$.
- If $f(x_1)>f(x_2)$, discard $[a,x_1)$.
- If equal, retain $[x_1,x_2]$ (or safely shrink symmetrically).

Each iteration reduces the interval by roughly one half, but the two new evaluations are usually not reusable. $\delta$ must be large enough that the values can be distinguished and small relative to the target interval accuracy.

## Golden-section search

Choose interior points so one survives after interval reduction. With
$$\tau=\frac{\sqrt5-1}{2}\approx0.618,$$
set $x_1=b-\tau(b-a)$ and $x_2=a+\tau(b-a)$. After the first iteration, only one new function evaluation is needed per reduction. The bracket length is multiplied by $\tau$ each time.

## Derivative bisection

If $f'$ is continuous and $f'(a)<0<f'(b)$, bisection applied to $f'(x)=0$ finds a stationary point. Under convexity, it is a global minimum. Without convexity, the sign transition suggests a local minimum but does not rule out others.

## Safeguarded strategy

Maintain a bracket and propose faster interpolation steps. Reject a proposal if it leaves the bracket, barely shrinks it, or uses unreliable curvature; take a golden/dichotomic step instead. This combines guaranteed bracket reduction with local speed.

If the interval length is multiplied by $\rho<1$, after $k$ steps it is $\rho^k(b-a)$, so
$$
k\ge \frac{\log(\varepsilon/(b-a))}{\log\rho}
$$
reaches width $\varepsilon$.

See [[Univariate Optimization]] and [[Interpolation Newton and Convexity]].


