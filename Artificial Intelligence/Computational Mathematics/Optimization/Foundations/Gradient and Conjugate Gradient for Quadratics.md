# Gradient and Conjugate Gradient for Quadratics

For $f(x)=\frac12x^TQx-b^Tx$ with $Q\succ0$, solving the optimization problem is equivalent to solving $Qx=b$.

## Steepest descent with exact line search

Let $g_k=Qx_k-b$. Then
$$
x_{k+1}=x_k-\alpha_kg_k,qquad
\alpha_k=\frac{g_k^Tg_k}{g_k^TQg_k}.
$$
Successive gradients are orthogonal under exact arithmetic. Yet on elongated ellipsoids the iterates zig-zag. In the $Q$-norm,
$$
\|x_{k+1}-x^*\|_Q
\le \frac{\kappa(Q)-1}{\kappa(Q)+1}\|x_k-x^*\|_Q.
$$
Large $\kappa$ therefore means slow convergence.

Useful stopping tests include $\|g_k\|\le\varepsilon_g$, relative residual $\|Qx_k-b\|/\|b\|$, small step, and small objective decrease. Do not rely only on objective change: a flat but nonoptimal region can fool it.

## Conjugate gradient (CG)

Directions are $Q$-conjugate when $d_i^TQd_j=0$ for $i\ne j$. CG builds such directions without storing all earlier ones:

1. $r_0=b-Qx_0$, $d_0=r_0$.
2. $\alpha_k=(r_k^Tr_k)/(d_k^TQd_k)$.
3. $x_{k+1}=x_k+\alpha_kd_k$.
4. $r_{k+1}=r_k-\alpha_kQd_k$.
5. $\beta_{k+1}=(r_{k+1}^Tr_{k+1})/(r_k^Tr_k)$.
6. $d_{k+1}=r_{k+1}+\beta_{k+1}d_k$.

In exact arithmetic CG reaches the solution in at most $n$ iterations, but its practical strength is earlier convergence when eigenvalues cluster. Each step needs one matrix-vector product, so sparse $Q$ is ideal.

## Preconditioning

Choose $M\succ0$ such that applying $M^{-1}$ is cheap and $M^{-1}Q$ has clustered eigenvalues. Preconditioned CG uses $z_k=M^{-1}r_k$ and replaces $r_k^Tr_k$ by $r_k^Tz_k$. A good preconditioner lowers iteration count without making an iteration too expensive.

<details>
<summary>Why conjugate directions solve the quadratic</summary>

If $d_0,\ldots,d_{n-1}$ are nonzero $Q$-conjugate directions, they are linearly independent and form a basis. Exact minimization along them determines each coefficient of $x^*-x_0$ independently because cross terms vanish under the $Q$ inner product. After all $n$ directions, the error is zero.

</details>

See the numerical linear algebra view in [[Conjugate Gradient and Preconditioning|Conjugate Gradient and Preconditioning]].


