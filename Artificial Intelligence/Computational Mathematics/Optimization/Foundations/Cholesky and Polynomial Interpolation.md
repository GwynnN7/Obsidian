# Cholesky and Polynomial Interpolation

## Direct solution of a positive definite quadratic

For $Q\succ0$, factor
$$Q=LL^T$$
with $L$ lower triangular. Solve $Ly=b$ by forward substitution, then $L^Tx=y$ by back substitution. The factorization costs about $n^3/3$ operations; each triangular solve costs $O(n^2)$.

Compared with iterative methods:

- Cholesky is predictable and highly accurate for dense/moderate systems.
- CG is attractive for huge sparse systems and can stop early.
- A factorization is especially valuable when many right-hand sides share the same $Q$.

Cholesky exists uniquely with positive diagonal exactly for symmetric positive definite matrices. In floating point, failure of a diagonal pivot can signal indefiniteness or severe numerical issues.

## Back substitution

For upper triangular $Ux=b$:
$$
x_i=\frac{b_i-\sum_{j=i+1}^nU_{ij}x_j}{U_{ii}},qquad i=n,n-1,\ldots,1.
$$
The analogous forward formula solves lower triangular systems.

## Polynomial interpolation as motivation

Given distinct nodes $t_i$ and values $y_i$, a polynomial of degree at most $n-1$ satisfying $p(t_i)=y_i$ is found from the Vandermonde system
$$
\begin{bmatrix}
1&t_1&\cdots&t_1^{n-1}\\
\vdots&\vdots&&\vdots\\
1&t_n&\cdots&t_n^{n-1}
\end{bmatrix}a=y.
$$
Distinct nodes make the matrix invertible, but monomial Vandermonde matrices can be badly conditioned. Existence of an algebraic solution does not imply reliable numerical computation. Prefer stable bases or formulations when possible.

<details>
<summary>Why the interpolating polynomial is unique</summary>

If two degree-$\le n-1$ polynomials agree at $n$ distinct nodes, their difference has $n$ roots. A nonzero polynomial of degree at most $n-1$ cannot have $n$ distinct roots, so the difference is identically zero.

</details>

See also [[Direct Methods for Linear Systems|Direct Methods for Linear Systems]].


