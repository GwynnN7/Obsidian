---
tags: [artificial-intelligence-fundamental, games]
---

# Alpha-Beta Pruning

Eliminate branches of the [[Minimax]] tree that cannot affect the final decision. **Same answer, fewer nodes.**

## Bounds

- **$\alpha$**: the largest value found so far - the best choice for **MAX** along the path to the root.
- **$\beta$**: the smallest value found so far - the best choice for **MIN** along the path to the root.

## Pruning condition

If a node's best possible value is already **worse** than a value found at the same depth or higher in a different branch, prune it.

## Efficiency

Can reduce the time complexity substantially, but the actual reduction depends on **move ordering** - best pruning requires the best moves to be examined first.

Related: [[Minimax]], [[Heuristic Game Search]].
