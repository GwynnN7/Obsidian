---
tags: [artificial-intelligence-fundamental, logic]
---

# Logical Inference

$$
KB \vdash_i \alpha
$$

deriving $\alpha$ from the KB using an inference algorithm $i$.

| Property | Meaning |
|---|---|
| **Soundness** | every derived sentence is entailed: $KB\vdash_i\alpha \Rightarrow KB\models\alpha$. *It only proves true things.* |
| **Completeness** | every entailed sentence can be derived: $KB\models\alpha \Rightarrow KB\vdash_i\alpha$. *It proves all the true things.* |

## Ways to prove something

- **Chain of conclusions** leading to the desired goal.
- **Model checking:** enumerate all the combinations and check if the goal is true - see [[Entailment and Models]].
- **Theorem proving via inference rules:** start from the KB and apply inference rules to derive the goal (or work backwards).
- **Theorem proving as search:** search for the optimal sequence of inference rules to derive the goal - the connection back to [[Search Problems]].

Related: [[Propositional Logic]], [[FOL Inference]].
