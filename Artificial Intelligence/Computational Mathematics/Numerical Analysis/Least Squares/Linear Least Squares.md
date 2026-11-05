# Linear Least Squares

For $A\in\mathbb R^{m\times n}$ and $b\in\mathbb R^m$, solve
$$\min_x\|Ax-b\|_2.$$
This covers inconsistent systems, linear regression, and polynomial/data fitting.

## Geometry and normal equations

At the solution, the residual $r=b-Ax$ is orthogonal to the column space:
$$A^T(b-Ax)=0,$$
so
$$A^TAx=A^Tb.$$
If $A$ has full column rank, $A^TA\succ0$ and the solution is unique.

## QR solution

With thin $A=QR$ and $Q^TQ=I$,
$$\|Ax-b\|_2^2=\|Rx-Q^Tb\|_2^2+\|(I-QQ^T)b\|_2^2.$$
Thus solve
$$Rx=Q^Tb.$$
Householder QR avoids forming $A^TA$ and is the standard reliable full-rank method.

## Why not normal equations by default

$\kappa_2(A^TA)=\kappa_2(A)^2$. Forming $A^TA$ can also lose information through rounding. Cholesky on the normal equations is fast, but QR is safer; SVD is safest for rank deficiency.

<details>
<summary>Projection proof of the least-squares condition</summary>

$Ax$ lies in $\operatorname{Im}(A)$. The closest point in a subspace is its orthogonal projection, so the difference $b-Ax$ must be perpendicular to every column of $A$. Collecting the dot products gives $A^T(b-Ax)=0$.

</details>

## Polynomial fitting

For data $(t_i,y_i)$, choose basis functions $\phi_j$ and set $A_{ij}=\phi_j(t_i)$. Scaling/centering $t_i$ and using a stable basis can drastically improve conditioning compared with high-degree monomials.

## Method choice

- Full rank, dense: QR.
- Suspected rank deficiency or need for minimum norm: SVD.
- Very large sparse: iterative least-squares methods (or normal-equation Krylov methods with care).

See [[Pseudoinverse and Regularization]].


