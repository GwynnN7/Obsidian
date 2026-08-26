---
tags: [artificial-intelligence-fundamental, csp]
---

# Tree-Structured CSPs

If the constraint graph is a **tree** (or a forest), the CSP is solvable in $O(nd^2)$ - **linear in the number of variables**.

## Linear-time algorithm

1. **Topological sort** of the graph, designating an arbitrary node as the root.
2. Enforce **arc consistency** on the tree, propagating from children to parent.
3. Assign each node a value consistent with its parent - **no backtracking is ever required**.

## Making a general CSP tree-shaped

Techniques exist to convert a general CSP into a tree structure, typically by identifying and **sequentially removing** a variable (a cutset) and enforcing consistency on the residual graph.

> [!note]
> Same lesson as in [[Bayesian Network Inference]], where exact inference is linear on a **polytree** and NP-hard in general. Structure, not size, is what makes these problems tractable.

Related: [[Constraint Propagation]], [[Constraint Satisfaction Problems]].
