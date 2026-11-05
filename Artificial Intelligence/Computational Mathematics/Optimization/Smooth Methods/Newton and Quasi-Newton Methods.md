# Newton and Quasi-Newton Methods

## Newton's method

Minimize the second-order model
$$m_k(d)=f(x_k)+g_k^Td+\frac12d^TH_kd.$$
When $H_k=\nabla^2f(x_k)\succ0$, its minimizer solves $H_kd_k=-g_k$, followed by $x_{k+1}=x_k+\alpha_kd_k$. Near a nondegenerate minimizer, the full step is accepted and convergence is quadratic.

Never form $H_k^{-1}$ explicitly; solve the system. If $H_k$ is indefinite, the raw Newton direction may not descend. Remedies include damping $H_k+\lambda I$, modifying a factorization, line search, or a trust region.

## Quasi-Newton secant condition

Let $s_k=x_{k+1}-x_k$ and $y_k=g_{k+1}-g_k$. Approximate Hessians satisfy $B_{k+1}s_k=y_k$. BFGS updates an inverse-Hessian approximation by
$$
H_{k+1}=(I-\rho_ks_ky_k^T)H_k(I-\rho_ky_ks_k^T)+\rho_ks_ks_k^T,
\qquad \rho_k=(y_k^Ts_k)^{-1}.
$$
If $H_k\succ0$ and $y_k^Ts_k>0$, positive definiteness is preserved. Wolfe line search helps enforce this condition.

## BFGS versus L-BFGS

- BFGS stores a dense $n\times n$ approximation and is strong for moderate $n$.
- L-BFGS stores only the latest $m$ pairs $(s_i,y_i)$ and applies them with a two-loop recursion, using $O(mn)$ memory/work.

| Method | Derivatives | Local speed | Per-step burden |
|---|---|---:|---:|
| Gradient | first | linear/sublinear | low |
| Newton | first + second | quadratic | Hessian and solve |
| BFGS | first | superlinear | dense $O(n^2)$ |
| L-BFGS | first | typically fast | $O(mn)$ |

<details>
<summary>Why Newton is quadratic locally</summary>

Newton solves $\nabla f(x)=0$. A Lipschitz Hessian and nonsingular $\nabla^2f(x^*)$ make the gradient linearization error quadratic; multiplication by a locally bounded inverse Hessian gives $\|x_{k+1}-x^*\|=O(\|x_k-x^*\|^2)$.

</details>


