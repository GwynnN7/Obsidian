# QR Factorization

For $A\in\mathbb R^{m\times n}$ with $m\ge n$, a thin QR factorization is
$$A=QR,$$
where $Q\in\mathbb R^{m\times n}$ has orthonormal columns and $R\in\mathbb R^{n\times n}$ is upper triangular. If $A$ has full column rank, $R$ is nonsingular.

## Householder reflectors

A reflector has
$$H=I-2\frac{vv^T}{v^Tv}.$$
It is symmetric and orthogonal. Choose $v$ so $Hx$ is a multiple of $e_1$, zeroing all entries of $x$ below the first. Applying successive reflectors to columns reduces $A$ to $R$; their product forms $Q^T$.

To avoid cancellation, choose the target sign opposite to $x_1$, e.g.
$$v=x+\operatorname{sign}(x_1)\|x\|_2e_1.$$
Store reflector vectors compactly below the diagonal rather than forming each full $H$.

## Algorithm and cost

For $k=1,\ldots,n$, build a reflector for $A_{k:m,k}$ and apply it to the trailing block. Dense thin QR costs about $2mn^2-\frac23n^3$ flops. Applying stored reflectors costs much less than explicitly forming $Q$.

## Why Householder QR

Classical Gram-Schmidt can lose orthogonality on nearly dependent columns. Modified Gram-Schmidt is better, but Householder QR is the standard stable dense method. Orthogonal transformations preserve 2-norm, so rounding errors are not magnified by the transformations themselves.

<details>
<summary>Why a reflector is orthogonal</summary>

Let $P=vv^T/(v^Tv)$, so $P^2=P$. Then $H^TH=H^2=(I-2P)^2=I-4P+4P^2=I$.

</details>

Main use: [[Linear Least Squares|Linear Least Squares]]. It also underlies [[Numerical Eigenvalue Methods|Numerical Eigenvalue Methods]].


