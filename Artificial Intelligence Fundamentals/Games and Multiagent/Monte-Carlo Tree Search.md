---
tags: [artificial-intelligence-fundamental, games]
---

# Monte-Carlo Tree Search

An alternative to [[Alpha-Beta Pruning]], useful when the branching factor is large **or no effective evaluation function is available** - so you cannot do [[Heuristic Game Search]].

## Ingredients

- **Rollout / playout:** simulate an entire game episode from a given state to obtain a final utility.
- **State value:** the **average** final utility across multiple rollouts.
- **Playout policy:** the function governing move selection during the simulation, often based on [[Reinforcement Learning]].

## The cycle

1. **Selection** - choose a path from the root using the selection policy.
2. **Expansion** - generate one new node from the selected path using the playout policy.
3. **Simulation** - a complete rollout from the new node.
4. **Back-propagation** - the result updates the value of all the nodes along the selection path, from the new node up to the root.
5. **Move decision** - play the move with the highest number of playouts, update the state, repeat.

## Selection policy: exploitation vs exploration

The policy must balance **exploiting** (playing out from good states) and **exploring** (playing out from less-visited states).

## UCT - Upper Confidence bounds applied to Trees

Choose the node maximizing **UCB1**:

$$
UCB1 = \underbrace{\frac{\text{total utility through }n}{\text{playouts through }n}}_{\text{exploitation}} + \underbrace{\text{exploration term}}_{\text{high when playouts through }n\text{ are few}}
$$

Benefit: ensures that the moves with the highest number of playouts also have the highest utility, while preventing premature convergence on seemingly good but poorly explored actions.

Related: [[Minimax]], [[Local Search]].
