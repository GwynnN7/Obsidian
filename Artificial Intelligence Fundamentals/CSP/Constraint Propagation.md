---
tags: [artificial-intelligence-fundamental, csp]
---

# Constraint Propagation

Inference: shrink the domains before (or instead of) searching.

## Forward checking

Applied immediately after a variable $X$ is assigned: prune the conflicting values from the domains of the neighbouring **unassigned** variables $Y$ connected by a constraint. If any domain becomes **empty**, backtrack immediately.

## Arc consistency (AC)

- $X$ is **arc-consistent** w.r.t. a binary constraint $X\to Y$ if, for **every** value in the domain of $X$, there exists at least one permissible value in the domain of $Y$.
- A constraint graph is arc-consistent when all its variables satisfy the condition.

## AC-3

A complete algorithm returning an arc-consistent graph, or failure.

1. Initialize a queue with all the arcs.
2. Select an arc $X\to Y$ and enforce arc consistency on $X$.
3. If any value is removed from $X$'s domain, add all the relevant **reverse arcs** $K\to X$ back to the queue.
4. Terminate when the queue is empty, or on a domain failure.

Complexity: $O(cd^3)$ for a CSP with $c$ binary arcs and domain size $d$.

## MAC - Maintaining Arc Consistency

Forward checking upgraded: run AC-3 after **every** assignment $X$, initializing the queue with the arcs $Y\to X$ for every unassigned neighbour $Y$. Backtrack if AC-3 detects an inconsistency.

Related: [[Backtracking Search]], [[Tree-Structured CSPs]].
