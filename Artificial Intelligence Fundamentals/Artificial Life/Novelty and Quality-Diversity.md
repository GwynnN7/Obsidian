---
tags: [artificial-intelligence-fundamental, alife]
---

# Novelty and Quality-Diversity

What to do when maximizing the objective **leads you away** from the solution - a deceptive optimization problem.

## Novelty-Based Search (NS)

Supports **open-ended evolution**, where the goal is survival and replication rather than task performance.

- **Mechanism:** a numerical encoding of *behaviours*, plus an **archive** of the novel behaviours already discovered.
- **Fitness is based solely on novelty** - sparseness in the behaviour space relative to the archive.
- **Sparsity metric:**
  $$\rho(x)=\frac1k\sum_{i=1}^{k}d(x,n_i)$$
  with $n_i$ the $i$-th nearest neighbour of $x$ in the behaviour space. If $\rho(x)>\rho_{min}$, add $x$ to the archive.

> [!note]
> That formula is a $k$-nearest-neighbour computation - see [[k-NN]]. The "metric defines everything" caveat applies here too: the behaviour characterization *is* the inductive bias.

## Quality-Diversity (QD)

Balance task-based fitness maximization with the generation of novel solutions (divergent optimization). Goal: a **maximal and diverse set of high-quality** solutions.

- **Behavioural Characterization (BC):** a vector describing an individual's path, used to measure novelty.
- **QD-score:** $\sum_{i=1}^{\#bins} Q_i$, with $Q_i$ the highest fitness found in behaviour bin $i$ - integrating quality and diversity in one number.

## MAP-Elites

- Maintains a **multidimensional archive of phenotypic elites**.
- The behaviour space is divided into a discrete **grid of bins**, so that the space of possible behaviours is equally valued.
- Random parents are selected from bins to generate an offspring; if the offspring's fitness beats the current occupant of its behaviour bin, it **replaces** the elite.
- **Sparsity mitigation:** hierarchical versions refine coarse-grained cells over time.

Related: [[Evolvability]], [[Evolutionary Algorithms]], [[Local Search]].
