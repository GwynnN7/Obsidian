---
tags: [artificial-intelligence-fundamental, models]
---

# Hidden Markov Models

## Components

- **State variables $X_t$:** hidden / unobservable.
- **Evidence variables $E_t$:** observable.

**Models:**

- **Transition model:** the state evolution as a Markov process, $P(X_t\mid X_{0:t-1})$.
- **Sensor / observation model** (sensor Markov assumption): the observation depends **only on the current hidden state**, since the state summarizes all the previous states and observations:
  $$P(E_t\mid X_{0:t},E_{1:t-1})\ \to\ P(E_t\mid X_t)$$

**Joint distribution:**

$$
P(X_{0:t},E_{1:t})=P(X_0)\prod_i P(X_i\mid X_{i-1})P(E_i\mid X_i)
$$

HMMs are a kind of [[Bayesian Networks|Bayesian network]] that can manage an unlimited number of variables over time. Typically a single discrete hidden variable with one or more observations.

## The four inference tasks

| Task | Query | Uses |
|---|---|---|
| **Filtering** (state estimation) | $P(X_t\mid e_{1:t})$ - the current belief state | past evidence |
| **Prediction** | $P(X_{t+k}\mid e_{1:t})$, $k>0$ | transition model only |
| **Smoothing** | $P(X_k\mid e_{1:t})$, $0\le k\le t-1$ | **all** evidence, past and future |
| **Most-likely explanation** | the state sequence with maximum likelihood of generating the evidence | Viterbi |

## Filtering

**Recursive** estimation: update the previous estimate with the current evidence $e_t$, so the whole history is not reprocessed at every step:

$$
P(X_t\mid e_{1:t})=f(e_t,\ P(X_{t-1}\mid e_{1:t-1}))
$$

**Forward message:** $f_{1:t}=\alpha\,\text{Forward}(f_{1:t-1},e_t)$ with $f_0=P(X_0)$.

**Complexity:** constant time and space **per step** $\to$ **linear** $O(t)$ overall.

## Prediction

For $k=1$ it is part of the filtering step. For $k\ge2$, recursively:

$$
P(X_{t+k}\mid e_{1:t})=\sum P(X_{t+k}\mid X_{t+k-1})P(X_{t+k-1}\mid e_{1:t})
$$

Only the transition model is involved - prediction incorporates no future observations.

**Asymptotic behaviour:** as $k\to\infty$, $P(X_{t+k}\mid e_{1:t})$ converges to the **stationary distribution** of the Markov process. The convergence rate (**mixing time**) is shorter when the transition model is more uncertain.

## Smoothing - the forward-backward algorithm

Uses a **forward** message (past evidence) **and** a **backward** message $b_{k+1:t}$ (future evidence) to improve the estimate of a past state $P(X_k\mid e_{1:t})$.

- $b_{k+1:t}=\text{backward}(b_{k+2:t},e_{k+1})$, initialized as $b_{t+1:t}=1$ (there is no $e_{t+1}$).
- The backward message costs $O(k)$, but smoothing **all** the states costs $O(t^2)$.

## Most-likely sequence - Viterbi

Find the path through the graph maximizing the joint likelihood.

- The optimal path at a given time is derived recursively from the optimal path at the preceding time, combined with the best next step.
- **Linear** time and space in $t$.

Related: [[Reasoning over Time]], [[Bayesian Network Inference]], [[RNN]].
