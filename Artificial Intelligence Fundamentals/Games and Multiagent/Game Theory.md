---
tags: [artificial-intelligence-fundamental, theory]
---

# Game Theory

## Normal form games

A model for a **single, simultaneous move**, where no player observes the others' actions.

Components:

- a set of $n$ **players**,
- a set of **actions** executable by each agent,
- a **payoff function** giving the utility each agent receives after the joint action.

In a two-player game the outcomes are shown as a **payoff matrix**.

## Strategies

- **Pure strategy:** a deterministic policy - in a normal-form game, the single action chosen.
- **Mixed strategy:** a randomized policy - actions selected according to a probability distribution.
- **Strategy profile:** the assignment of a strategy to every player.

## Dominance

- **Strong dominance:** strategy $s$ is strictly better than $s'$ **regardless** of the other players' choices.
- **Weak dominance:** $s$ is at least as good as $s'$ for all the others' choices, and strictly better for at least one.
- **Dominant strategy:** dominates every other available strategy.
- **Rationality principle:** a rational agent selects the dominant strategy or, minimally, avoids dominated ones - assuming the others do the same.
- **Dominant strategy equilibrium:** everyone plays their dominant strategy, and nobody has an incentive to deviate.

## Nash equilibrium

Dominant strategies are rare in practice, which makes the Nash equilibrium the widely applicable concept.

> A strategy profile where **no player can unilaterally change their strategy and get a higher payoff**, assuming the others keep theirs.

- **Best response:** each player's strategy is optimal given the others' fixed choices. A dominant strategy is the best response to *any* choice.
- **Stability:** a locally stable point - no immediate incentive to deviate.
- **Existence:** guaranteed to exist for all normal-form games **when mixed strategies are allowed**.
- **Coordination challenge:** with multiple Nash equilibria and no communication, the choice falls back on local decision-making.

## Finding equilibria

Exhaustive search over all strategy profiles: check whether any player benefits from a different profile; if not, it is a Nash equilibrium. Complexity $O(\#\text{actions}^{\#\text{players}})$.

Related: [[Social Welfare]], [[Multiagent Systems]], [[Adversarial Search]].
