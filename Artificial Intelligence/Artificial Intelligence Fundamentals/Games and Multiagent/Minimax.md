---
tags: [artificial-intelligence-fundamental, algorithms]
---

# Minimax

Determines the optimal strategy for each player, conceived as a **conditional plan**: what MAX should do given MIN's potential moves. Applicable to search in non-deterministic environments, where the opponent's move is modelled as a probability distribution.

## Values

- **Terminal nodes:** $MINIMAX(s)=UTILITY(s;p)$, the utility from MAX's perspective.
- **Intermediate nodes:** the utility for MAX **assuming optimal play** from that node onwards.

## Procedure (DFS)

1. DFS from the root down to the terminal states, obtaining their utility values.
2. Backtrack computing the values of the intermediate nodes:
   - **MIN** node $\to$ take the **minimum** among the children,
   - **MAX** node $\to$ take the **maximum**.

## Properties

- **Complete** on finite game trees.
- **Optimal** assuming perfect play from the opponent.
- Time $O(b^m)$, space $O(bm)$ - like DFS ([[Uninformed Search]]).

Related: [[Alpha-Beta Pruning]], [[Heuristic Game Search]], [[Stochastic Games]].
