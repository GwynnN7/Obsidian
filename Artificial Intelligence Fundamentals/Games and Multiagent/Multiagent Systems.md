---
tags: [artificial-intelligence-fundamental, theory]
---

# Multiagent Systems

## Single decision maker

- Multi-action scenarios requiring **coordination** to achieve common goals.
- Can be modelled as a **single agent** if control and sensors are centralized (one actor with many body parts).
- Often relies on the **benevolent agent assumption**: agents strictly follow instructions.

## Multiple decision makers

- Each agent has its own preferences and executes its own plan.
- **Coordination problem:** decision makers share the same overarching goal.
- **Competitive behaviour:** agents have opposing preferences.
- **[[Game Theory]]** is the framework for strategic decision-making, where each player explicitly accounts for the behaviour of others (cooperative vs non-cooperative games).

## Concurrency in planning

Agents' actions may execute simultaneously.

- **Interleaved execution:** eliminate concurrency by executing plans sequentially, one atomic action at a time. Leads to an **exponential** number of possible resulting plans in the number of actors and actions, and the plan must be correct across **all** interleavings.
- **True concurrency:** keep only a **partial ordering** among actions.
- **Perfect synchronization (global lock):** all actions take the same duration; in each time step every agent executes its action simultaneously. **NoOp** actions let an agent wait for the result of another. The **joint action** is the simultaneous action of all agents - which gives a large transition model, mitigated by assuming independence.

## Cooperation and coordination

- **Agent autonomy:** each agent independently chooses its own plan.
- **Convention adoption:** constrain plan selection *a priori* to pre-empt conflicts.
- **Communication:** establish common knowledge of a joint plan.
- **Plan recognition:** infer the plan of another agent - e.g. in swarms.

Related: [[Game Theory]], [[Advanced Planning]], [[Environments]].
