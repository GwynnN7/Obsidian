---
tags: [artificial-intelligence-fundamental, algorithms]
---

# Heuristic Game Search

$O(b^m)$ is usually intractable, so:

- **Type A strategy:** explore up to a fixed depth (**cut-off**) and use a heuristic function to evaluate the utility of the resulting nodes.
- **Type B strategy:** use a heuristic to dynamically **filter** moves, pursuing only the promising ones as far as possible.

Heuristics can be designed by exploiting combinations of specific game features, or **learned** from played positions - see [[Supervised Learning]].

> [!warning] Horizon effect
> The heuristic falsely evaluates a position as good because a critical negative outcome (e.g. a forced loss) occurs just **beyond** the fixed search depth.

Related: [[Minimax]], [[Alpha-Beta Pruning]], [[Monte-Carlo Tree Search]], [[Informed Search]].
