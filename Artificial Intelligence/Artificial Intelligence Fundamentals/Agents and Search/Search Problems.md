---
tags: [artificial-intelligence-fundamental, theory]
---

# Search Problems

A **search algorithm** finds a sequence of actions defining a path to a goal state.

- **Informed vs Uninformed:** is there knowledge about the distance to the goal? ([[Informed Search]] vs [[Uninformed Search]])
- **Open loop (offline) vs Closed loop (online):** compute the entire optimal action sequence before execution, or re-evaluate the environment continuously during the action phase.

## Problem classes

| Class | Solution shape |
|---|---|
| **Deterministic, fully observable** | a single-state problem: a fixed **sequence of actions** |
| **Non-observable (sensorless)** | a sequence of actions with no sensor feedback - see [[Advanced Planning\|sensorless planning]] |
| **Non-deterministic / partially observable** | a **contingency plan** or **policy**, structured as a decision tree - see [[Planning in Non-Deterministic Environments]] |
| **Unknown state space** | an **exploration** problem: discover, and possibly discard, parts of the state space |

Related: [[Tree Search]], [[Environments]].
