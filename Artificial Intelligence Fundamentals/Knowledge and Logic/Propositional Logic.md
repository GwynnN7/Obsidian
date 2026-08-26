---
tags: [artificial-intelligence-fundamental, logic]
---

# Propositional Logic

Atomic propositional symbols ($P_1,P_2,\dots$) connected by logical connectives $\lnot,\wedge,\vee,\Rightarrow,\Leftrightarrow$.

## Inference by enumeration

Sound and complete:

1. Generate all the possible truth assignments for all the symbols in the KB and $\alpha$.
2. Check which models satisfy the KB.
3. Verify that $\alpha$ is true in all of them, $M(KB)\subseteq M(\alpha)$.

**Complexity $O(2^n)$** with $n$ symbols - a complexity shared by all the known general inference algorithms for PL.

## Conjunctive Normal Form (CNF)

Any PL statement can be converted to a **conjunction of clauses**, where a clause is a **disjunction of literals**:

$$
(A\vee\lnot B)\wedge(B\vee C\vee\lnot D)
$$

## Resolution

A **sound and complete** inference rule, used for proof by refutation.

**Procedure:** convert $(KB\wedge\lnot\alpha)$ to CNF, apply the resolution rule repeatedly to generate new clauses; if the **empty clause** is derived, there is a contradiction, hence $KB\models\alpha$.

## Horn clauses

Clauses with **at most one positive literal**.

> [!important]
> If the KB is in Horn form, forward and backward chaining (using **Modus Ponens**) are **linear** in time - instead of $O(2^n)$. Restricting the language buys tractability.

### Forward chaining (FC) - data-driven

Start from the known facts and apply rules to derive new conclusions until the goal is generated. **Sound and complete** for Horn KBs.

1. Collect all the symbols in the KB that are true.
2. Repeat: for all Horn clauses, determine if the body holds; if so, add the head to the KB.
3. Return true when $\alpha$ is generated, false if nothing new can be derived.

### Backward chaining (BC) - goal-directed

Start from the query and recursively work backwards to find supporting premises.

1. Start from the query; return true if it is a fact.
2. Otherwise check all the rules having $q$ as head.
3. If any of their bodies is true, return true; else iterate.

Can cost **less** than forward chaining, since it only touches what is relevant to the goal.

Related: [[First-Order Logic]], [[Logical Inference]], [[Entailment and Models]], [[Concept Learning]] (the same language, used as a hypothesis space).
