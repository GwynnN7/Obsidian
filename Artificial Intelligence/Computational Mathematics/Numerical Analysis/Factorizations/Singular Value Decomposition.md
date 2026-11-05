# Singular Value Decomposition

Every $A\in\mathbb R^{m\times n}$ has an SVD
$$A=U\Sigma V^T,$$
where $U$ and $V$ are orthogonal and singular values satisfy
$$\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_p\ge0,\qquad p=\min(m,n).$$

Geometrically, $V^T$ rotates/refects the input, $\Sigma$ scales orthogonal axes, and $U$ rotates/refects the output. The image of the unit sphere is an ellipsoid with semiaxes $\sigma_i$.

## Thin and compact forms

For $m\ge n$, the thin SVD keeps $U\in\mathbb R^{m\times n}$, $\Sigma\in\mathbb R^{n\times n}$, and $V\in\mathbb R^{n\times n}$. The compact SVD keeps only the $r=\operatorname{rank}(A)$ positive singular values and associated vectors.

## Relations to eigenvalues

$$A^TA=V\Sigma^2V^T,qquad AA^T=U\Sigma^2U^T.$$
Thus singular values are square roots of eigenvalues of $A^TA$, but computing the SVD through $A^TA$ can square the condition number and lose accuracy.

## Four fundamental subspaces

With rank $r$:

- $\operatorname{Im}(A)=\operatorname{span}(u_1,\ldots,u_r)$;
- $\ker(A^T)=\operatorname{span}(u_{r+1},\ldots,u_m)$;
- $\operatorname{Im}(A^T)=\operatorname{span}(v_1,\ldots,v_r)$;
- $\ker(A)=\operatorname{span}(v_{r+1},\ldots,v_n)$.

## Consequences

$$\|A\|_2=\sigma_1,\qquad \operatorname{rank}(A)=\#\{i:\sigma_i>0\}.$$
For full column rank, $\kappa_2(A)=\sigma_1/\sigma_n$. Very small singular values identify directions in which inversion amplifies noise.

See [[Matrix Norms and Low-Rank Approximation]] and [[Pseudoinverse and Regularization|Pseudoinverse and Regularization]].


