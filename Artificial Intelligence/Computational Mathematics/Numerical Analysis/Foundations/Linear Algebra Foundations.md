# Linear Algebra Foundations

## Matrices as transformations

For $A=[a_1\ \cdots\ a_n]\in\mathbb R^{m\times n}$,
$$Ax=\sum_{j=1}^na_jx_j.$$
Thus $\operatorname{Im}(A)$ is the span of the columns, while $\ker(A)=\{x:Ax=0\}$. Rank is the dimension of the image and satisfies rank-nullity:
$$\operatorname{rank}(A)+\dim\ker(A)=n.$$

Matrix multiplication is associative but not commutative. Parenthesization matters: $A(Bv)$ costs $O(n^2)$ for square matrices, while forming $(AB)v$ costs $O(n^3)$.

## Linear systems

$Ax=b$ is solvable iff $b\in\operatorname{Im}(A)$. For square $A$, the following are equivalent:

- $A$ is invertible;
- $\ker(A)=\{0\}$;
- $\operatorname{rank}(A)=n$;
- every $b$ has a unique solution;
- $0$ is not an eigenvalue.

Do not compute $A^{-1}$ merely to solve $Ax=b$; a factorization and triangular solves are cheaper and more stable.

## Structured matrices

- Diagonal: solves and products cost $O(n)$.
- Triangular: solve by substitution in $O(n^2)$.
- Orthogonal: $Q^TQ=I$, so $Q^{-1}=Q^T$ and lengths are preserved.
- Symmetric: $A=A^T$, with real eigenvalues and orthonormal eigenvectors.
- Sparse: store and operate only on nonzeros; ordering and fill-in matter.

## Computational complexity

A flop is one arithmetic operation. Big-$O$ hides constants and lower-order terms, but constants, memory movement, and sparsity matter in practice. A dense matrix needs $O(n^2)$ memory, which becomes limiting before $O(n^3)$ arithmetic for very large $n$.

## Floating point reminder

Binary64 has machine precision around $2.22\times10^{-16}$. Individual operations are modeled as
$$\operatorname{fl}(a\circ b)=(a\circ b)(1+\delta),\qquad |\delta|\lesssim u,$$
when no overflow/underflow occurs. Cancellation can destroy relative accuracy even when each operation is correctly rounded.

Continue with [[Orthogonality Eigenvalues and Positive Definiteness]].


