# Pseudoinverse and Regularization

Let $A=U\Sigma V^T$ have rank $r$. The Moore-Penrose pseudoinverse is
$$A^+=V\Sigma^+U^T
=\sum_{i=1}^r\frac1{\sigma_i}v_iu_i^T.$$

$x^+=A^+b$ is a least-squares solution; among all least-squares solutions it has minimum Euclidean norm. If the system is consistent, it is the minimum-norm exact solution.

## Why small singular values are dangerous

In SVD coordinates,
$$x^+=\sum_{i=1}^r\frac{u_i^Tb}{\sigma_i}v_i.$$
Noise components along $u_i$ are amplified by $1/\sigma_i$. A tiny nonzero singular value can therefore make the nominal solution meaningless.

## Truncated SVD

Keep only $\sigma_i\ge\tau$:
$$x_{TSVD}=\sum_{i:\sigma_i\ge\tau}\frac{u_i^Tb}{\sigma_i}v_i.$$
This removes unstable directions but introduces bias. The threshold should reflect noise and validation, not machine precision alone.

## Tikhonov/ridge regularization

Solve
$$
\min_x\|Ax-b\|_2^2+\lambda\|x\|_2^2,
$$
giving
$$
(A^TA+\lambda I)x=A^Tb.
$$
In SVD coordinates the filter factor is $\sigma_i/(\sigma_i^2+\lambda)$ rather than $1/\sigma_i$, smoothly damping small singular directions.

## Choosing regularization

Larger $\lambda$ lowers variance/noise amplification but increases bias. Choose it using known noise, cross-validation, discrepancy principles, or an L-curve. Always scale variables before interpreting an isotropic penalty.

<details>
<summary>Minimum-norm property</summary>

Every least-squares solution is $x=x^++z$ with $z\in\ker(A)$. The SVD shows $x^+\in\operatorname{Im}(A^T)=\ker(A)^\perp$. Hence $\|x\|^2=\|x^+\|^2+\|z\|^2$, minimized at $z=0$.

</details>


