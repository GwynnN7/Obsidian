# Conditioning and Stability

Conditioning belongs to the **problem**; stability belongs to the **algorithm**.

## Condition numbers

For nonsingular $A$,
$$
\kappa(A)=\|A\|\|A^{-1}\|,
\qquad \kappa_2(A)=\frac{\sigma_{\max}}{\sigma_{\min}}.
$$
To first order, perturbations in $b$ satisfy
$$
\frac{\|\delta x\|}{\|x\|}\lesssim
\kappa(A)\frac{\|\delta b\|}{\|b\|}.
$$
A large condition number means some input directions are greatly amplified. It does not by itself say the algorithm is bad.

For least squares, sensitivity depends on $\kappa(A)$ and also on the residual/angle between $b$ and $\operatorname{Im}(A)$. Near rank deficiency is dangerous.

## Forward and backward error

- Forward error: distance from computed $\hat x$ to exact $x$.
- Backward error: smallest perturbation to the data for which $\hat x$ is exact.

A backward stable algorithm returns the exact answer to a nearby problem. Forward error is then roughly condition number times backward error.

## Residual-based estimate

With $r=b-A\hat x$ and $e=x-\hat x$, $Ae=r$, so under compatible scaling
$$
\frac{\|e\|}{\|x\|}\le
\kappa(A)\frac{\|r\|}{\|b\|}.
$$
A tiny residual can coexist with a large error when $\kappa(A)$ is huge.

## Stability of core methods

- Householder QR is backward stable because it uses orthogonal transformations.
- Triangular substitution is backward stable under standard assumptions.
- LU with partial pivoting is usually backward stable, with a bound involving element growth.
- Normal equations are risky because $\kappa(A^TA)=\kappa(A)^2$ and forming $A^TA$ adds rounding error.
- SVD exposes numerical rank and is the most reliable expensive option.

## Cancellation and scaling

Subtracting nearly equal numbers loses relative digits. Stable formulas reformulate expressions to avoid it. Scale rows, columns, variables, and units so magnitudes are comparable; scaling can improve both conditioning and floating-point behavior.

## Practical workflow

1. Estimate or understand conditioning.
2. Choose a structure-aware stable method.
3. Compute a residual and backward-error measure.
4. Use conditioning to estimate forward accuracy.
5. If accuracy is insufficient, improve precision, scaling, regularization, or formulation - not merely the stopping tolerance.

<details>
<summary>Why normal equations square the condition number</summary>

The eigenvalues of $A^TA$ are $\sigma_i(A)^2$. Hence
$$\kappa_2(A^TA)=\frac{\sigma_{\max}^2}{\sigma_{\min}^2}=\kappa_2(A)^2.$$

</details>


