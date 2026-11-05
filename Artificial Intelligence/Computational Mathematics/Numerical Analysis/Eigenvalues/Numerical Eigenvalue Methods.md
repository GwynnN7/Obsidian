# Numerical Eigenvalue Methods

## QR algorithm

Starting with $A_0=A$, repeat
$$A_k=Q_kR_k,\qquad A_{k+1}=R_kQ_k=Q_k^TA_kQ_k.$$
The matrices are orthogonally similar and have the same eigenvalues. Under suitable conditions, $A_k$ approaches triangular (diagonal for symmetric $A$), revealing eigenvalues.

Practical QR first reduces $A$ to upper Hessenberg form (tridiagonal if symmetric) using Householder transformations. Each QR step then costs $O(n^2)$ rather than $O(n^3)$. Shifts accelerate convergence, and deflation splits off converged eigenvalues when subdiagonal entries become negligible.

## Arnoldi/Lanczos approximation

Arnoldi forms $H_k=Q_k^TAQ_k$. Eigenpairs $(\theta,y)$ of $H_k$ give Ritz pairs
$$\theta,\qquad u=Q_ky.$$
The residual is cheap:
$$\|Au-\theta u\|_2=|h_{k+1,k}e_k^Ty|.$$
Lanczos is the symmetric specialization and efficiently approximates extreme eigenvalues.

## Power and inverse ideas

Power iteration converges toward a dominant eigenvector when one eigenvalue has strictly largest magnitude. Inverse iteration applies $(A-\mu I)^{-1}$ and converges toward an eigenvalue near shift $\mu$; factorize $A-\mu I$ once when reusing the shift. Rayleigh quotient iteration adapts the shift and is very fast near a symmetric eigenpair.

## What to use

- All eigenvalues of a dense matrix: shifted QR.
- A few extreme eigenvalues of a huge sparse symmetric matrix: Lanczos.
- A few eigenvalues of a huge nonsymmetric matrix: Arnoldi.
- Singular values: use SVD algorithms, not generic eigenvalue computation on $A^TA$ when accuracy matters.

See [[QR Factorization|QR Factorization]] and [[Krylov Subspaces and Arnoldi|Krylov Subspaces and Arnoldi]].


