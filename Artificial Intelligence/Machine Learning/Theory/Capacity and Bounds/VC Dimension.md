---
tags: [machine-learning, theory]
---

# VC Dimension

Vapnik-Chervonenkis dimension: a measure of the **capacity** (expressive power, flexibility) of a class of hypotheses, valid also for infinite $H$.

Not $|H|$, but the number of distinct instances that $H$ can completely discriminate.

## Shattering

- $X$: instance space, $N$ = number of instances in it.
- Binary classification $\to$ there are $2^N$ possible **dichotomies** (labelings of the $N$ points with $-1/+1$).
- A dichotomy is *represented* in $H$ if there exists $h\in H$ that realizes it.
- **$H$ shatters $X$** iff $H$ can represent all the possible dichotomies on $X$ (0 errors).

## Definition

$VC(H)$ = the maximum cardinality of a set (configuration) of points in $X$ that can be shattered by $H$.

$VC(H)=p$ means:

- $H$ shatters **at least one** configuration of $p$ points, and
- $H$ cannot shatter **any** configuration of $p+1$ points.

If arbitrarily large finite sets can be shattered, $VC$ is infinite.

## Example: lines in $\mathbb{R}^2$

- $VC(H)\ge3$: there is a configuration of 3 points shattered by lines. Note that *not all* configurations of 3 points work (3 aligned points do not) - it is enough that **one** does.
- $VC(H)<4$: with 4 points you can always draw six lines between pairs; two of them cross; put the points joined by the crossing lines in the same class and they are not linearly separable (the XOR again).
- So $VC(H)=3$.

## General results

- Linear separators (LTU) in $\mathbb{R}^n$: $VC(H)=n+1$. This is an upper bound - constraining the model can lower it.
- Lines in $\mathbb{R}$: 2. Intervals on $\mathbb{R}$: 2.
- Circles centred in the origin in 2D, $\operatorname{sign}(x\cdot x-b)$: 1; $\operatorname{sign}(kx\cdot x-b)$: 2.
- Circles in 2D (free radius and centre): 3. Circles with fixed radius and free centre: 3.
- [[k-NN]]: infinite.

## VC vs number of parameters

Related, but **not** the same:

- the circle examples above have the same VC with a different number of free parameters;
- redundant parameters can always be added;
- there exist models with one parameter and infinite VC-dim.

For NN, capacity depends on the number of units *and* on the size of the weights - see [[Neural Networks]].

Related: [[Statistical Learning Theory]], [[Structural Risk Minimization]].
