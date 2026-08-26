---
tags: [artificial-intelligence-fundamental, planning]
---

# Planning Heuristics

Heuristics for planning come from **relaxing** the problem: solve an easier version and use its cost as an estimate. Same idea as [[Informed Search]], where admissibility is what buys optimality.

## Ignore-precondition heuristic

Assume **every action is applicable**.

- Keep from each action's effect only the literals present in the goal.
- Compute the minimum number of actions whose combined effects satisfy the goal.
- **Admissible** $\to$ A* stays optimal; greedy algorithms find approximate solutions.

## Ignore-delete-list heuristic

**Remove the negative literals** from the action effects.

- Guarantees **monotonic** progress towards the goal, since no action can undo a previous step.
- Hill-climbing search works well with it - see [[Local Search]].

## Domain-independent pruning

- **Symmetry reduction:** prune all but one of the symmetric branches of the search tree - a large reduction of the search space.
- **Forward pruning:** prune based on a **preferred action**. Obtain a relaxed plan solving a relaxed version of the problem; a preferred action is either a step of the relaxed plan or an action achieving one of its preconditions.
  > [!warning]
  > Forward pruning might prune away the optimal solution.

## Reducing the state space

- **State abstraction:** a many-to-one mapping from states to an abstract representation - remove some fluents to get an approximate solution (fewer states, fewer steps), then add them back if possible to recover the exact one.
- **Decomposition:** divide the problem into independent parts, solve each, combine the plans.
  - **Subgoal independence assumption:** the cost of a conjunction of subgoals is approximated by the sum of the costs of solving each independently.
  > [!warning]
  > **Not admissible:** it can give a longer solution than the exact one, if two subplans contain actions that could have been merged.

Related: [[Classical Planning]], [[Informed Search]].
