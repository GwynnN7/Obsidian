# Course Coverage Checklist

This is a compact audit of the taught material. Each item links to the note where it is explained.

## Optimization

- Problem statement, infimum versus minimum, local/global/$\varepsilon$-optimal solutions: [[Optimization Problems]].
- Feasibility, numerical tolerances, reformulations, boxes, domains, extended values, Pareto ideas: [[Optimization Problems]].
- Linear, separable, homogeneous and nonhomogeneous quadratic models: [[Elementary Optimization Models]].
- Symmetric part, eigenvalue classification, positive (semi)definiteness, existence and uniqueness: [[Quadratic Optimization]].
- Gradient method on quadratics, exact line search, condition-number rate, stopping: [[Gradient and Conjugate Gradient for Quadratics]].
- Conjugate directions, CG recurrence, finite termination, preconditioning: [[Gradient and Conjugate Gradient for Quadratics]].
- Backsolve, Cholesky cost/reuse, polynomial interpolation and Vandermonde warning: [[Cholesky and Polynomial Interpolation]].
- Global versus local scalar optimization, Lipschitz lower bounds, unimodality, bracketing: [[Univariate Optimization]].
- Dichotomic, golden-section and derivative bisection: [[Bracketing and Dichotomic Search]].
- Parabolic interpolation, Newton, safeguards, scalar convexity: [[Interpolation Newton and Convexity]].
- Directional derivatives, differentiability, gradient geometry, Jacobian/chain rule, Hessian: [[Multivariate Derivatives]].
- Taylor models and first-/second-order optimality: [[Taylor Models and Optimality Conditions]].
- Convexity tests, strong convexity, global consequences: [[Convex Functions]].
- Neural networks and backpropagation as the motivating nonconvex model: [[Neural Networks as Optimization Models]].
- General descent framework and scaled directions: [[Smooth Unconstrained Optimization]].
- Armijo-Wolfe and fixed-step convergence under Lipschitz gradients: [[Line Search and Fixed Steps]].
- Newton, indefinite curvature, BFGS, L-BFGS: [[Newton and Quasi-Newton Methods]].
- Nonlinear CG, heavy ball and accelerated gradient: [[Momentum and Accelerated Gradient]].

Not included because it was outside the supplied slide sequence: nonsmooth optimization, constrained optimality/duality, and constrained algorithms.

## Numerical Analysis

- Matrix transformations/products, subspaces, rank, systems, structured matrices, complexity and floating point: [[Linear Algebra Foundations]].
- Inner products, projections, eigenvalues, symmetric/SPD matrices, Rayleigh quotient: [[Orthogonality Eigenvalues and Positive Definiteness]].
- Full/thin/compact SVD and four fundamental subspaces: [[Singular Value Decomposition]].
- Induced/Frobenius norms, Eckart-Young, compression, denoising and PCA: [[Matrix Norms and Low-Rank Approximation]].
- Householder reflectors, cancellation-safe sign, thin QR and cost: [[QR Factorization]].
- Triangular solves, LU, pivoting and Cholesky: [[Direct Methods for Linear Systems]].
- Sparse storage, graph view, fill-in, ordering and direct/iterative tradeoff: [[Sparse Linear Systems]].
- Regression/fitting, geometry, normal equations and QR solve: [[Linear Least Squares]].
- SVD least squares, pseudoinverse, minimum norm, TSVD and Tikhonov: [[Pseudoinverse and Regularization]].
- Krylov spaces, Arnoldi decomposition, Lanczos and polynomial viewpoint: [[Krylov Subspaces and Arnoldi]].
- GMRES/MINRES, residual minimization, restart and convergence: [[GMRES and MINRES]].
- CG derivation, rate, left/right/symmetric preconditioning: [[Conjugate Gradient and Preconditioning]].
- QR eigenvalue algorithm, shifts/deflation, Ritz pairs and sparse eigensolvers: [[Numerical Eigenvalue Methods]].
- Linear-system and least-squares conditioning, forward/backward stability, residual bounds, stability of QR/solves, normal-equation warning: [[Conditioning and Stability]].

