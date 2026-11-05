# Taylor Models and Optimality Conditions

For smooth $f$, near $x$:
$$
f(x+d)=f(x)+\nabla f(x)^Td+\frac12d^T\nabla^2f(x)d+o(\|d\|^2).
$$

## First-order condition

If $x^*$ is an unconstrained local minimum and $f$ is differentiable, then $\nabla f(x^*)=0$. This is necessary, not sufficient. A point satisfying it is stationary.

## Second-order conditions

At a stationary point:

- Necessary: $\nabla^2f(x^*)\succeq0$.
- Sufficient for a strict local minimum: $\nabla^2f(x^*)\succ0$.
- A negative Hessian eigenvalue rules out a local minimum.
- A semidefinite but singular Hessian is inconclusive; higher-order terms matter.

Examples: $x^4$ has a strict minimum at $0$ with zero Hessian; $x^3$ has zero gradient and Hessian at $0$ but no minimum.

<details>
<summary>Proof of the first-order condition</summary>

For any direction $d$, define $\phi(t)=f(x^*+td)$. A local minimum at $t=0$ gives $\phi'(0)=0$. Differentiability gives $\phi'(0)=\nabla f(x^*)^Td$. Since this holds for every $d$, choose $d=\nabla f(x^*)$ to obtain $\|\nabla f(x^*)\|^2=0$.

</details>

<details>
<summary>Why a positive definite Hessian is sufficient</summary>

Continuity and $H(x^*)\succ0$ imply that in a small neighborhood $d^TH(x)d\ge m\|d\|^2$ for some $m>0$. Taylor's theorem yields $f(x^*+d)\ge f(x^*)+\frac m2\|d\|^2$ for sufficiently small nonzero $d$.

</details>

For convex functions, stationarity is globally sufficient; see [[Convex Functions]].


