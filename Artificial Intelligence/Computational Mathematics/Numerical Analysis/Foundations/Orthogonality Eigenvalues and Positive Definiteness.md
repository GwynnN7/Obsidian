# Orthogonality, Eigenvalues, and Positive Definiteness

## Inner products and norms

The Euclidean inner product is $x^Ty$ and $\|x\|_2=\sqrt{x^Tx}$. Cauchy-Schwarz says $|x^Ty|\le\|x\|_2\|y\|_2$. Vectors are orthogonal when $x^Ty=0$.

An orthogonal matrix $Q$ preserves inner products and norms:
$$\|Qx\|_2=\|x\|_2.$$
This is why orthogonal transformations are numerically attractive: they do not amplify errors in 2-norm.

The orthogonal projector onto the columns of $Q$ with $Q^TQ=I$ is $P=QQ^T$. It satisfies $P^2=P=P^T$.

## Eigenvalues

$Av=\lambda v$ with $v\ne0$. A general matrix may lack a real orthogonal eigenbasis, but a real symmetric matrix has
$$A=Q\Lambda Q^T,$$
with real $\Lambda$ and orthogonal $Q$ (spectral theorem).

## Positive definiteness

For symmetric $A$:

- $A\succ0$ iff $x^TAx>0$ for all $x\ne0$ iff all eigenvalues are positive.
- $A\succeq0$ iff all eigenvalues are nonnegative.
- $A\succ0$ implies invertible and admits a Cholesky factorization.

$A^TA\succeq0$ because $x^TA^TAx=\|Ax\|_2^2$. It is positive definite exactly when $A$ has full column rank.

## Rayleigh quotient

For symmetric $A$,
$$R_A(x)=\frac{x^TAx}{x^Tx}$$
lies between $\lambda_{\min}$ and $\lambda_{\max}$. Its stationary values are eigenvalues. This connects eigenvalue approximation, optimization, and CG convergence.

<details>
<summary>Spectral-theorem consequence for quadratic forms</summary>

Writing $x=Qz$ gives $x^TAx=z^T\Lambda z=\sum_i\lambda_i z_i^2$. The sign of every quadratic form is therefore determined exactly by the signs of the eigenvalues.

</details>

See [[Singular Value Decomposition|Singular Value Decomposition]] and [[Direct Methods for Linear Systems|Direct Methods for Linear Systems]].


