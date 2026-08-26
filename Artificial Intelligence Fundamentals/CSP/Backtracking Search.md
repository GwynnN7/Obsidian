---
tags: [artificial-intelligence-fundamental, csp]
---

# Backtracking Search

Reduce the branching factor by imposing a **fixed order** on the variables, which limits the search to $O(d^n)$ leaves. It is essentially [[Uninformed Search|DFS]] where each node assigns a single variable.

## Variable ordering (fail-first)

- **MRV - Minimum Remaining Values:** pick the variable with the **fewest remaining legal values**, to prune impossible branches as early as possible.
- **Degree heuristic:** tie-breaker - pick the variable involved in the **most constraints**, to maximize constraint propagation early.

## Value ordering (least-constraining)

- **LCV - Least-Constraining Value:** pick the value that **eliminates the fewest options** for the neighbouring variables, preserving future flexibility.

> [!note]
> Variable ordering wants to **fail fast** (find the dead end now), value ordering wants to **succeed** (keep options open). Not a contradiction: you choose the variable most likely to fail, then the value least likely to.

## Local search alternative

- **Complete-state formulation:** every state is a complete (but possibly inconsistent) assignment.
- Start from a **random** assignment of all variables.
- **Min-conflicts heuristic:** reassign the variable whose change resolves the largest number of broken constraints.

See [[Local Search]].

Related: [[Constraint Propagation]], [[Constraint Satisfaction Problems]].
