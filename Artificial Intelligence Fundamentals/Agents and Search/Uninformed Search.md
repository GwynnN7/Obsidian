---
tags: [artificial-intelligence-fundamental, search]
---

# Uninformed Search

No knowledge of the distance to the goal - only the structure of the problem. Metrics as in [[Tree Search]]; $b$ = branching factor, $d$ = depth of the shallowest solution, $m$ = max depth.

## BFS - Breadth-First Search

Expand layer by layer, **FIFO** frontier.

- Complete if $b$ is finite.
- **Optimal only if the cost per step is uniform** (cost = 1).
- Time and space $O(b^d)$ - exponential in both, and memory is the killer.

## Dijkstra (uniform-cost)

Extends BFS by expanding the **least-cost** node first; the frontier is a priority queue.

- Optimal and complete if all action costs are positive.

## DFS - Depth-First Search

Expand the **deepest** unexpanded node, **LIFO** frontier.

- **Not** optimal.
- Time $O(b^m)$, but space only **linear**, $O(b\times m)$ - its whole selling point.
- Complete in finite spaces only if loop avoidance is implemented.

Related: [[Informed Search]], [[Tree Search]].
