# GMRES and MINRES

## GMRES

For a general nonsymmetric system $Ax=b$, GMRES chooses $x_k\in x_0+\mathcal K_k(A,r_0)$ to minimize the residual 2-norm. With Arnoldi,
$$
r_k=b-A(x_0+Q_ky),
$$
and the large problem reduces to
$$
\min_y\|\beta e_1-\bar H_ky\|_2,
\qquad \beta=\|r_0\|_2.
$$
Update the small least-squares factorization using Givens rotations rather than solving from scratch.

The residual norm is nonincreasing in exact arithmetic. But basis storage and orthogonalization grow with $k$, so restarted GMRES($m$) discards the basis every $m$ steps. Restarting saves memory but can destroy convergence information and cause stagnation.

## MINRES

For symmetric (possibly indefinite) $A$, MINRES minimizes the residual using the Lanczos recurrence. It has short recurrences and lower memory than full GMRES. Do not use CG on an indefinite matrix; MINRES is the appropriate symmetric alternative.

## Convergence viewpoint

GMRES residuals satisfy
$$
\|r_k\|_2=\min_{p\in\mathcal P_k,\ p(0)=1}\|p(A)r_0\|_2.
$$
Eigenvalue clustering helps for normal matrices, but for nonnormal matrices eigenvalues alone may poorly predict convergence.

## Practical details

- Stop using a relative residual consistent with the problem scale.
- Monitor the true residual occasionally; recurrence-based estimates can drift.
- Preconditioning is usually essential.
- A small residual does not imply a small forward error for ill-conditioned $A$.

See [[Krylov Subspaces and Arnoldi]] and [[Conjugate Gradient and Preconditioning]].


