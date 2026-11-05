# Quadratic Optimization

Consider
$$f(x)=\frac12x^TQx-b^Tx+c,$$
with $Q=Q^T$.

## Definiteness controls the problem

- $Q\succ0$ (all eigenvalues positive): $f$ is strictly convex and has the unique minimizer $x^*=Q^{-1}b$.
- $Q\succeq0$: $f$ is convex. A minimizer exists iff $b\in\operatorname{Im}(Q)$; then all solutions of $Qx=b$ are optimal. Otherwise $f$ is unbounded below along a nullspace direction.
- $Q$ indefinite: a negative-curvature direction $d$ with $d^TQd<0$ makes $f(x+td)\to-\infty$.
- $Q\preceq0$: minimization on all $\mathbb R^n$ is generally unbounded unless the function is constant in every relevant direction.

When $Q\succ0$, completing the square in matrix form gives
$$
f(x)-f(x^*)=\frac12(x-x^*)^TQ(x-x^*)=\frac12\|x-x^*\|_Q^2.
$$

## Spectral geometry

Write $Q=U\Lambda U^T$. In coordinates $z=U^Tx$, the quadratic is a sum of independent scalar quadratics. Level sets are ellipsoids. The condition number
$$\kappa_2(Q)=\frac{\lambda_{\max}}{\lambda_{\min}}$$
measures elongation and strongly affects iterative methods.

## Exact line minimization

Along $x+\alpha d$,
$$
\phi(\alpha)=f(x+\alpha d),\qquad
\alpha^*=-\frac{\nabla f(x)^Td}{d^TQd}
$$
when $d^TQd>0$. With steepest descent $d=-g$, this becomes
$$\alpha^*=\frac{g^Tg}{g^TQg}.$$

<details>
<summary>Proof of the positive definite solution</summary>

Since $Q\succ0$, $Qx=b$ has the unique solution $x^*$. Substitute $b=Qx^*$ and expand:
$$
f(x)-f(x^*)=\frac12x^TQx-x^TQx^*+\frac12{x^*}^TQx^*
=\frac12(x-x^*)^TQ(x-x^*).
$$
This is positive for every $x\ne x^*$.

</details>

Connections: [[Gradient and Conjugate Gradient for Quadratics]], [[Cholesky and Polynomial Interpolation]], and [[Conditioning and Stability|Conditioning and Stability]].


