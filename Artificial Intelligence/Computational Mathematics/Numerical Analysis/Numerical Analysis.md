# Numerical Analysis

This module focuses on solving
$$Ax=b\qquad\text{and}\qquad \min_x\|Ax-b\|_2$$
quickly and accurately, often for large matrices. The central lesson is that a mathematically correct formula can still be a bad numerical algorithm.

## Course map

1. [[Linear Algebra Foundations|Linear Algebra Foundations]]
2. [[Orthogonality Eigenvalues and Positive Definiteness|Orthogonality, Eigenvalues, and Positive Definiteness]]
3. [[Singular Value Decomposition|Singular Value Decomposition]] and [[Matrix Norms and Low-Rank Approximation|Matrix Norms and Low-Rank Approximation]]
4. [[QR Factorization|QR Factorization]]
5. [[Direct Methods for Linear Systems|Direct Methods for Linear Systems]] and [[Sparse Linear Systems|Sparse Linear Systems]]
6. [[Linear Least Squares|Linear Least Squares]] and [[Pseudoinverse and Regularization|Pseudoinverse and Regularization]]
7. [[Krylov Subspaces and Arnoldi|Krylov Subspaces and Arnoldi]], [[GMRES and MINRES|GMRES and MINRES]], and [[Conjugate Gradient and Preconditioning|Conjugate Gradient and Preconditioning]]
8. [[Numerical Eigenvalue Methods|Numerical Eigenvalue Methods]]
9. [[Conditioning and Stability|Conditioning and Stability]]

## Cost vocabulary

- Dense matrix-vector product: $O(mn)$.
- Dense $n\times n$ matrix product: about $2n^3$ flops.
- Dense factorization: typically $O(n^3)$.
- Triangular solve: $O(n^2)$.
- Sparse matrix-vector product: $O(\operatorname{nnz}(A))$.

Always distinguish problem sensitivity (**conditioning**) from error introduced by the algorithm (**stability**).


