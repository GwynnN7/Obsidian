# Direct Methods for Linear Systems

## Solve easy structure first

Diagonal systems cost $O(n)$. Triangular systems cost $O(n^2)$ by substitution. Orthogonal systems solve as $x=Q^Tb$. Factorizations reduce a difficult matrix to these easy cases.

## LU and Gaussian elimination

Gaussian elimination produces
$$PA=LU,$$
where $P$ is a permutation, $L$ unit lower triangular, and $U$ upper triangular. Solve $Ax=b$ through:

1. permute: $Pb$;
2. forward solve $Ly=Pb$;
3. backward solve $Ux=y$.

Dense LU costs about $\frac23n^3$ flops; the two solves cost $O(n^2)$. Reuse the factors for multiple right-hand sides.

## Partial pivoting

At column $k$, swap in the largest available $|a_{ik}|$ as pivot. This avoids division by a tiny pivot and controls element growth. LU with partial pivoting is backward stable for most practical matrices, although specially constructed worst cases exist.

Do not use elimination without pivoting for a general matrix. A nonzero but tiny pivot can generate huge multipliers and catastrophic rounding error.

## Symmetric positive definite systems

If $A\succ0$, use Cholesky:
$$A=LL^T.$$
It costs about $n^3/3$, half of LU, uses half the storage, and needs no pivoting in exact arithmetic. Solve $Ly=b$, then $L^Tx=y$.

## Factorization choice

| Matrix | Preferred direct method |
|---|---|
| general square | pivoted LU |
| symmetric positive definite | Cholesky |
| rectangular least squares | QR |
| rank-deficient / diagnostic accuracy | SVD |

## Residual versus error

A small residual $r=b-A\hat x$ does not guarantee a small forward error when $A$ is ill-conditioned. Use [[Conditioning and Stability|Conditioning and Stability]] to interpret it.


