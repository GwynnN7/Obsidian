---
tags: [artificial-intelligence-fundamental, search]
---

# Tree Search

Expand successor states from already visited states.

- The **frontier** is the set of unexpanded nodes: it separates the expanded from the unreached.
- Key problem: **nodes are not unique**, so the search tree can be infinite unless you avoid cycles or remember all reached states.

## The four metrics

Every algorithm is judged on:

| | |
|---|---|
| **Completeness** | is a solution guaranteed to be found if one exists? |
| **Time complexity** | number of nodes generated |
| **Space complexity** | max number of nodes stored in memory |
| **Optimality** | does it find the lowest-cost solution? |

Related: [[Uninformed Search]], [[Informed Search]], [[Search Problems]].
