---
tags: [artificial-intelligence-fundamental, alife]
---

# Evolutionary Strategies

## Black-Box Optimization (BBO)

Maximize an objective $f:S\to\mathbb{R}$ over a search space $S$.

- **Derivative-free:** solutions are found by **sampling** points from the search space.
- Performance metrics: the number of evaluations needed to converge, and the value of the final solution.

## ES

Stochastic BBO for **continuous** functions.

- **Individual:** phenotypic/decision parameters $x_0$ **plus** a set of *evolvable* strategy parameters (e.g. the mutation rate itself).
- **Mutation:** add random gaussian noise, which explores the search space uniformly.
- **Recombination:** defined by the **mixing number** $p$ - how many randomly chosen parents generate one offspring. $p=1$ is cloning.

## Selection schemes

| | Who competes | Focus |
|---|---|---|
| $(\mu+\lambda)$ | parents **and** offspring | **exploitation** |
| $(\mu,\lambda)$ | only offspring | **exploration** - avoids over-fitting, good for deceptive problems |

If $\mu=\lambda$, $(\mu,\lambda)$-ES degenerates into a **random walk** (no selection at all), whereas $(\mu+\lambda)$-ES still selects.

## CMA-ES

*Covariance Matrix Adaptation ES.* Uses an **adaptive covariance matrix** that self-adjusts to the local curvature of the fitness landscape.

Population sampled from a multivariate gaussian $\mathcal{N}(m_g,C_g)$, initialized with $C_0=I$ and random $m_g\in\mathbb{R}^n$:

$$
x_k^{g+1}=m_g+\sigma\,\mathcal{N}(0,C_g)\quad \forall k=1,\dots,\lambda
\qquad\Longleftrightarrow\qquad
x^g\sim\mathcal{N}(m_g,\sigma^2C_g)
$$

**Mean update** by truncated selection ($\mu<\lambda$) over the offspring sorted by increasing fitness, with normalized weights $w_i$:

$$
m_{g+1}=\sum_{i=1}^{\mu} w_i\,x_i^{g+1},
\qquad w_1\ge w_2\ge\dots\ge w_\mu>0,\quad \sum_i w_i=1 .
$$

> [!note]
> Adapting a covariance matrix to the local curvature is the evolutionary analogue of second-order methods in [[Gradient Descent]] - same intuition, no derivatives.

Related: [[Evolutionary Algorithms]], [[Local Search]].
