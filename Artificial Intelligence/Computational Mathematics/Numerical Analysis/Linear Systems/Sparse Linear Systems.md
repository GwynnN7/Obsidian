# Sparse Linear Systems

A sparse matrix has few nonzeros relative to $mn$. Store it in formats such as CSR/CSC rather than dense arrays. Matrix-vector multiplication then costs $O(\operatorname{nnz}(A))$.

## Sparse direct methods

Sparse LU/Cholesky can create **fill-in**: entries that were zero become nonzero in the factors. Fill determines time and memory more than the original nonzero count. Permuting rows/columns before factorization can reduce fill; common strategies approximate minimum degree or use graph separators.

The sparsity pattern is a graph:

- vertices correspond to variables;
- edges correspond to nonzeros (for symmetric matrices);
- elimination connects neighbors, explaining fill-in.

## Direct versus iterative

Sparse direct methods are robust and good for multiple right-hand sides, but memory can explode. Krylov methods only need matrix-vector products and a few vectors, making them better for huge systems, provided a useful preconditioner exists.

## Matrix-free computation

An iterative method may accept an operation $v\mapsto Av$ without storing $A$. This is crucial when $A$ is implicitly defined by a simulation, graph, or derivative operator.

## Complexity warning

Sparse does not automatically mean easy. Runtime depends on pattern, conditioning, ordering, and preconditioning. A matrix with $O(n)$ nonzeros can still have dense factors.

See [[Krylov Subspaces and Arnoldi|Krylov Subspaces and Arnoldi]] and [[Conjugate Gradient and Preconditioning|Conjugate Gradient and Preconditioning]].


