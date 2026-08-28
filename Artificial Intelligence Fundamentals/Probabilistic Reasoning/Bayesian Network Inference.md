---
tags: [artificial-intelligence-fundamental, algorithms]
---

# Bayesian Network Inference

**Goal:** compute the posterior $P(Y\mid e)$, with $Y$ the query variables and $E$ the evidence variables with observed assignment $e$.

Multiple query variables decompose: $P(Y_1,Y_2\mid e)=P(Y_2\mid Y_1,e)P(Y_1\mid e)$.

## Exact inference

**NP-hard in general.** But:

- **Polytrees** (singly-connected networks, where any two nodes are connected by at most one undirected path): **linear** time.
- **Clustering** algorithms (e.g. Junction Tree) group nodes into "meganodes" to make exact inference possible.

> [!note]
> Same structural story as [[Tree-Structured CSPs]]: the shape of the graph, not its size, decides tractability.

### Inference by enumeration

The network structure lets us compute the joint efficiently, and inference becomes systematic **sums of products** of conditional probabilities:

- **sums** correspond to looping over variable assignments,
- **products** correspond to combining conditional distributions.

### Variable elimination

A refinement of enumeration: perform the summations **right to left**, marginalizing variables out as early as possible, to avoid recomputing the same subexpressions.

- A space-time trade-off: intermediate results (**factors**) are stored.
- **Factors** are matrices that are functions of the involved variables, e.g. $f(A)$.
- **Pointwise product:** $f(X_i,Y_j)\times g(Y_j,Z_k)$ gives a new factor $h(X_i,Y_j,Z_k)$.

## Approximate inference

### Direct sampling

Draw $N$ samples from the network's sampling distribution to compute an approximate posterior $Q$. $Q$ is a **consistent estimator**: it converges to the true posterior as $N\to\infty$.

### Rejection sampling

To estimate $P(X=x\mid e)$: generate samples from the network and **reject** any sample not matching the evidence $e$; the estimate is the frequency of $X=x$ among the accepted samples.

> [!warning]
> The convergence rate depends heavily on $P(e)$: rare evidence means almost everything is rejected.

Related: [[Bayesian Networks]], [[Reasoning over Time]].
