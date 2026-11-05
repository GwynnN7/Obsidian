# Krylov Subspaces and Arnoldi

Given $A$ and starting vector $r_0$, the $k$-th Krylov subspace is
$$
\mathcal K_k(A,r_0)=\operatorname{span}\{r_0,Ar_0,A^2r_0,\ldots,A^{k-1}r_0\}.
$$
Krylov methods seek an approximate solution in $x_0+\mathcal K_k$. They need only products $Av$, so they work with sparse or matrix-free operators.

## Arnoldi orthogonalization

Arnoldi builds an orthonormal basis $Q_k=[q_1,\ldots,q_k]$:

1. $q_1=r_0/\|r_0\|$.
2. Compute $v=Aq_k$.
3. For $j=1,\ldots,k$, set $h_{jk}=q_j^Tv$ and $v\leftarrow v-h_{jk}q_j$.
4. $h_{k+1,k}=\|v\|$ and $q_{k+1}=v/h_{k+1,k}$.

The Arnoldi relation is
$$AQ_k=Q_{k+1}\bar H_k,$$
where $\bar H_k$ is upper Hessenberg. This compresses the action of a huge $A$ to a small matrix.

Modified Gram-Schmidt and possible reorthogonalization protect orthogonality in finite precision. If $h_{k+1,k}=0$, the Krylov space is invariant; this is a happy breakdown if the desired solution/eigeninformation is captured.

## Symmetric case: Lanczos

For symmetric $A$, Arnoldi reduces to a three-term recurrence and $H_k$ is symmetric tridiagonal. Lanczos therefore uses less storage and work, but loss of orthogonality can create duplicate-looking Ritz values; reorthogonalization may be needed.

## Polynomial viewpoint

Every vector in $\mathcal K_k$ has the form $p_{k-1}(A)r_0$. Convergence depends on how well low-degree polynomials can be small on unwanted parts of the spectrum while satisfying normalization.

Applications: [[GMRES and MINRES]], [[Conjugate Gradient and Preconditioning]], and [[Numerical Eigenvalue Methods|Numerical Eigenvalue Methods]].


