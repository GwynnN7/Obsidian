---
tags: [artificial-intelligence-fundamental, agents]
---

# Environments

The environment defines the context of the agent's operation.

| Dichotomy | Meaning |
|---|---|
| **Static vs Dynamic** | Static: predictable and observable, the optimal solution is fixed - which often produces *fragile* agents. Dynamic: needs continuous adaptation. |
| **Fully vs Partially observable** | Can the sensors access the complete state? |
| **Episodic vs Sequential** | Sequential: past actions influence future states and rewards. Episodic: actions are independent. |
| **Deterministic vs Stochastic** | Deterministic: an action always leads to the same state. Stochastic: uncertainty. |
| **Discrete vs Continuous** | Granularity of the state and action spaces. |
| **Single-agent vs Multi-agent** | How many interacting agents. |

These choices decide which techniques apply: [[Classical Planning]] assumes discrete, deterministic, static and fully observable; [[Planning in Non-Deterministic Environments]] and [[Probability and Decision Theory]] relax that.

Related: [[Rational Agents]].
