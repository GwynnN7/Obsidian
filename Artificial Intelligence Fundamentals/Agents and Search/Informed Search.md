---
tags: [artificial-intelligence-fundamental, search]
---

# Informed Search

Uses a **heuristic function** $h(n)$: an estimate of the minimum cost from node $n$ to the goal.

**Best-first search** = expand the node deemed most desirable according to an evaluation function.

## Greedy search

Select the node with the lowest $h(n)$.

- Non-optimal and non-complete (loops); complete only in a finite space with a repeat-check.

## A*

$$
f(n)=g(n)+h(n)
$$

combining the path cost from the start $g(n)$ with the heuristic estimate $h(n)$.

> [!important] Admissibility
> $h(n)$ is **admissible** if it never **overestimates** the actual cost to the goal. A* is complete and **optimal** if the heuristic is admissible.

- Time complexity is exponential **in the error of the heuristic**: the better the heuristic, the cheaper the search.

## Where heuristics come from

They can be designed by relaxing the problem (see [[Planning Heuristics]]) or **learned** from experience, e.g. by exploiting combinations of specific features - see [[Supervised Learning]].

Related: [[Uninformed Search]], [[Local Search]], [[Minimax]].
