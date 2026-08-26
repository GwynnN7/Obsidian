---
tags: [artificial-intelligence-fundamental, agents]
---

# Agent Architectures

From simple to complex:

| Architecture | Mechanism | Limit |
|---|---|---|
| **Table agent** | a comprehensive lookup table for all possible percepts | severely limited by the complexity of environments - a real effectiveness/efficiency trade-off |
| **Simple reflex agent** | condition-action rules only | linear time complexity, but viable only in simple environments |
| **Model-based agent** | builds an **internal state** estimating the current environment, capturing the transition model *and* the sensor model | |
| **Goal-based agent** | uses [[Search Problems\|search]] and [[Classical Planning\|planning]] to maximize immediate or expected long-term reward | |
| **Utility-based agent** | assigns an intrinsic **utility** value to intermediate states - finer than plain goal attainment | see [[Probability and Decision Theory]] |
| **Learning-based agent** | any of the above **plus** the ability to modify its performance by learning from context, guided by an external critic or by problem generation | see [[Reinforcement Learning]], [[Supervised Learning]] |

> [!note]
> The table agent is the AI version of the argument in [[Concept Learning]]: a lookup table has no bias and therefore does not generalize. Every step down this list adds structure that lets the agent handle situations it has not seen.

Related: [[Rational Agents]], [[Environments]].
