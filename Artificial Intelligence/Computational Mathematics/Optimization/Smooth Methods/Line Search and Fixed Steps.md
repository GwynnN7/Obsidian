# Line Search and Fixed Steps

## Armijo sufficient decrease

Given descent direction $d_k$, the Armijo condition is
$$
f(x_k+\alpha d_k)\le f(x_k)+c_1\alpha\nabla f(x_k)^Td_k,
\qquad 0<c_1<1.
$$
Backtracking starts from a trial $\alpha$ and repeatedly multiplies it by $\rho\in(0,1)$ until Armijo holds.

Armijo alone can accept steps that are too short. Wolfe adds
$$
\nabla f(x_k+\alpha d_k)^Td_k\ge c_2\nabla f(x_k)^Td_k,
\qquad c_1<c_2<1.
$$
Strong Wolfe bounds the absolute value of the new directional derivative. Wolfe searches are important for quasi-Newton curvature updates.

## Fixed steps for smooth functions

If the gradient is $L$-Lipschitz,
$$\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|,$$
the descent lemma gives
$$
f(x-\alpha g)\le f(x)-\alpha\left(1-\frac{L\alpha}{2}\right)\|g\|^2.
$$
Thus $0<\alpha<2/L$ guarantees descent; $\alpha=1/L$ is standard.

For convex $L$-smooth $f$, gradient descent has $f(x_k)-f^*=O(1/k)$. If $f$ is also $\mu$-strongly convex, convergence is linear and depends on $L/\mu$.

<details>
<summary>Descent lemma derivation</summary>

Integrate the gradient along the segment from $x$ to $y$ and use Lipschitz continuity:
$$f(y)\le f(x)+\nabla f(x)^T(y-x)+\frac L2\|y-x\|^2.$$
Set $y=x-\alpha\nabla f(x)$ to obtain the stated inequality.

</details>

## Practical rules

- Scale variables so coordinates have comparable effects.
- Reuse the previous accepted step as the next initial trial.
- Do not take an unverified full step on a nonconvex problem.
- Count function/gradient evaluations, not only iterations.

See [[Smooth Unconstrained Optimization]] and [[Gradient and Conjugate Gradient for Quadratics|Gradient and Conjugate Gradient for Quadratics]].


