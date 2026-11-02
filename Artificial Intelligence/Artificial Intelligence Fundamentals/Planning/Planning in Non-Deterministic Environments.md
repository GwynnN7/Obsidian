---
tags: [artificial-intelligence-fundamental, theory]
---

# Planning in Non-Deterministic Environments

When the environment is non-deterministic the solution is **not a sequence** but a **conditional plan** - a decision tree accommodating the possible environmental responses.

- The agent maintains a **belief state**: the set of all possible future states resulting from a current state-action pair.
- **AND-OR search trees** are used for conditional planning:
  - **OR** nodes = the agent's choices,
  - **AND** nodes = the environment's possible outcomes (all of them must be handled).

Related: [[Search Problems]], [[Advanced Planning]], [[Environments]], [[Probability and Decision Theory]].
