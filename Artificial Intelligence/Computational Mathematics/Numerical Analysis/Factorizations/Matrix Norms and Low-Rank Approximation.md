# Matrix Norms and Low-Rank Approximation

## Matrix norms

The induced 2-norm is
$$\|A\|_2=\max_{x\ne0}\frac{\|Ax\|_2}{\|x\|_2}=\sigma_1.$$
Other common induced norms are maximum absolute column sum $\|A\|_1$ and maximum absolute row sum $\|A\|_\infty$. The Frobenius norm is
$$\|A\|_F=\sqrt{\sum_{ij}|A_{ij}|^2}=\sqrt{\sum_i\sigma_i^2}.$$

Induced norms are submultiplicative: $\|AB\|\le\|A\|\|B\|$. Orthogonal multiplication preserves 2- and Frobenius norms.

## Truncated SVD

Let
$$A_k=\sum_{i=1}^k\sigma_i u_iv_i^T.$$
This is a rank-$k$ representation. Storage drops from $mn$ to about $k(m+n)$ when $k\ll\min(m,n)$.

The Eckart-Young theorem gives
$$
\min_{\operatorname{rank}(B)\le k}\|A-B\|_2=\sigma_{k+1},
$$
$$
\min_{\operatorname{rank}(B)\le k}\|A-B\|_F=\sqrt{\sum_{i>k}\sigma_i^2}.
$$

<details>
<summary>Eckart-Young proof idea</summary>

The truncated SVD attains the stated errors directly. For any rank-$k$ $B$, its kernel intersects the $(k+1)$-dimensional span of $v_1,\ldots,v_{k+1}$ in a nonzero unit vector $z$. Then $Bz=0$ and $\|(A-B)z\|=\|Az\|\ge\sigma_{k+1}$, proving optimality in 2-norm. The Frobenius result follows from orthogonal invariance and the unavoidable tail energy.

</details>

## Applications

- Compression and denoising: discard components below the noise scale.
- PCA: after centering data, right singular vectors give principal directions and $\sigma_i^2$ gives variance up to normalization.
- Latent semantic/image models: low-rank factors expose dominant patterns.

Choose $k$ using spectral decay, explained variance, validation, and noise level - not just a visually convenient cutoff.


