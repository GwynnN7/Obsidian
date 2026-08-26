---
tags: [artificial-intelligence-fundamental, search]
---

# Local Search

Aim for a *satisfactory* solution while keeping only a single (or a few) candidate solutions, improving them incrementally. The path is irrelevant - only the final state matters.

## Hill climbing (greedy local search)

Pick the action giving the largest immediate improvement.

- **Limit:** gets trapped in **local maxima**.
- **Mitigations:** random restarts, allowing sideways moves.

## Simulated annealing

Hill climbing that also permits random (suboptimal) moves, with the probability and magnitude decreasing over time, governed by a **temperature** parameter and an **annealing schedule**.

## Local beam search

Keep $k$ candidate solutions at once and expand the top $k$ states.

- Risk: **lack of diversity** - the $k$ states collapse onto the same region.
- Fix: **stochastic beam search**, selecting randomly with probability based on the solution value.

> [!note] Same problem, other name
> This is the [[Multiple Minima]] story: a non-convex landscape, local optima, random restarts, and noise used deliberately to escape them. [[Gradient Descent]] is hill climbing with a gradient instead of an enumeration of moves; [[Evolutionary Strategies]] are population-based local search without derivatives.

Related: [[Informed Search]], [[Constraint Satisfaction Problems]] (min-conflicts), [[Evolutionary Algorithms]].
