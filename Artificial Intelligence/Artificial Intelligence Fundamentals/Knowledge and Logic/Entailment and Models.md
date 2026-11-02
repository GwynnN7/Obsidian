---
tags: [artificial-intelligence-fundamental, theory]
---

# Entailment and Models

## Possible worlds / models

A **model** is a mathematical abstraction assigning a fixed truth value to every sentence.

- A model $m$ **satisfies** a sentence $\alpha$ if $\alpha$ is true in $m$.
- $M(\alpha)$ = the set of all models satisfying $\alpha$.

## Entailment

$$
\alpha \models \beta
$$

a **semantic** relation: $\beta$ logically follows from $\alpha$ **iff every model in which $\alpha$ is true also makes $\beta$ true**, i.e. $M(\alpha)\subseteq M(\beta)$. This implies $\alpha$ is a *stronger* assertion than $\beta$.

**Model checking:** enumerate all the possible models and check whether $\alpha$ is true in all the models where the KB is true, $M(KB)\subseteq M(\alpha)$.

## Properties of sentences

| | |
|---|---|
| **Logical equivalence** $\alpha\equiv\beta$ | iff $\alpha\models\beta$ **and** $\beta\models\alpha$ |
| **Validity** | true in **all** possible models |
| **Satisfiability** | true in **at least one** model |
| **Unsatisfiability** | true in **no** model |

## Proof by contradiction (reductio ad absurdum)

$$
KB\models\alpha \iff (KB\wedge\lnot\alpha) \text{ is unsatisfiable}
$$

The satisfiability problem (**SAT**) is NP-complete - which is why this is both the standard technique and the source of the cost.

Related: [[Logical Inference]], [[Propositional Logic]], [[Knowledge-Based Agents]].
