---
tags: [artificial-intelligence-fundamental, theory]
---

# Classical Planning

Finding a **sequence of actions** to accomplish a goal.

**Environment:** discrete, deterministic, static, fully observable - see [[Environments]]. Relaxing these gives [[Planning in Non-Deterministic Environments]] and [[Advanced Planning]].

Solved by problem-solving agents via [[Search Problems|search]], or logically via inference ([[Propositional Logic]], [[First-Order Logic]]).

## Algorithms

### Forward state-space search

- Start at the initial state.
- **Unify** the current state against the preconditions of each action schema to find the applicable actions.
- Apply the resulting substitutions to get a **ground action**.
- Apply the action to reach the next state.

### Backward search (goal-directed)

- Start at the **goal** state.
- Consider the **relevant** actions: those whose effect unifies with a goal literal and **none of whose effects negates any part of the goal**.
- Apply relevant actions until the sequence reaches the initial state.
- The **branching factor is generally lower**, since only relevant paths are considered.

Related: [[PDDL]], [[Planning Heuristics]].
