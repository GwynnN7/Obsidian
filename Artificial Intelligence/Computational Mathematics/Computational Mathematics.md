# Computational Mathematics

This course connects two viewpoints:

- [[Optimization|Optimization]] asks how to minimize a function and how to recognize an optimum.
- [[Numerical Analysis|Numerical Analysis]] asks how to solve the linear-algebra problems that appear inside optimization, accurately and efficiently.

## Suggested study order

1. Optimization: [[Optimization Problems|Optimization Problems]] -> [[Quadratic Optimization|Quadratic Optimization]] -> [[Univariate Optimization|Univariate Optimization]] -> [[Multivariate Derivatives|Multivariate Derivatives]] -> [[Smooth Unconstrained Optimization|Smooth Unconstrained Optimization]].
2. Numerical Analysis: [[Linear Algebra Foundations|Linear Algebra Foundations]] -> [[Singular Value Decomposition|Singular Value Decomposition]] -> [[QR Factorization|QR Factorization]] -> [[Direct Methods for Linear Systems|Direct Methods for Linear Systems]] -> [[Linear Least Squares|Linear Least Squares]] -> [[Krylov Subspaces and Arnoldi|Krylov Subspaces and Arnoldi]] -> [[Conditioning and Stability|Conditioning and Stability]].

## Shared ideas

- A positive definite quadratic $\frac12x^TQx-b^Tx$ is minimized by solving $Qx=b$.
- Condition numbers explain both slow gradient descent and sensitivity of linear systems.
- Conjugate gradient is both an optimization method and a Krylov linear solver.
- Least squares is an optimization problem whose reliable numerical solution needs QR or SVD.

Use [[Course Coverage Checklist]] to audit the material against the taught PDFs.



