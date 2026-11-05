# Conjugate Gradient and Preconditioning

CG solves $Ax=b$ when $A$ is symmetric positive definite. It is also minimization of $\phi(x)=\frac12x^TAx-b^Tx$.

## Algorithm

Initialize $r_0=b-Ax_0$, $p_0=r_0$. Repeat:
$$
\alpha_k=\frac{r_k^Tr_k}{p_k^TAp_k},\qquad
x_{k+1}=x_k+\alpha_kp_k,
$$
$$
r_{k+1}=r_k-\alpha_kAp_k,
$$
$$
\beta_{k+1}=\frac{r_{k+1}^Tr_{k+1}}{r_k^Tr_k},\qquad
p_{k+1}=r_{k+1}+\beta_{k+1}p_k.
$$

Residuals are mutually orthogonal and search directions are $A$-conjugate in exact arithmetic. CG is optimal over the current Krylov affine space in the $A$-norm.

## Convergence

$$
\|x_k-x^*\|_A\le
2\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^k\|x_0-x^*\|_A.
$$
This bound can be pessimistic; clustered eigenvalues often give much faster convergence.

## Preconditioning

Choose $M\succ0$ approximating $A$ but easy to solve. PCG uses $z_k=M^{-1}r_k$ and
$$
\alpha_k=\frac{r_k^Tz_k}{p_k^TAp_k},\qquad
\beta_{k+1}=\frac{r_{k+1}^Tz_{k+1}}{r_k^Tz_k}.
$$
Examples include diagonal/Jacobi scaling, incomplete Cholesky, and problem-specific multilevel methods.

For GMRES, left preconditioning minimizes the preconditioned residual $M^{-1}(b-Ax)$; right preconditioning solves $AM^{-1}y=b$, $x=M^{-1}y$, retaining the true residual interpretation. Symmetric preconditioning is required to preserve CG theory.

## Finite precision

CG may take more than $n$ steps in floating point because orthogonality is lost. Recompute the true residual periodically when high accuracy matters. Never apply standard CG unless symmetry and positive definiteness are justified.


