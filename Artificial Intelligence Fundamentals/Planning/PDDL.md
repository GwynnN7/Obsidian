---
tags: [artificial-intelligence-fundamental, planning]
---

# PDDL

*Planning Domain Definition Language* - a Lisp-based language for specifying planning problems (classical, continuous, partially-observable or multi-agent).

**Semantics:** database semantics, with the **closed-world** and **unique-names** assumptions.

Each program defines a **domain** of problems and can represent a combinatorial number of actions with a single schema. The output is a **plan**: a sequence of potentially parallelizable actions.

## States

A conjunction of **ground atomic fluents**:

- **Ground:** no variables.
- **Atomic:** a single literal, no structure.
- **Fluents:** aspects of the world that change over time.

## Actions (action schema)

- action **name**,
- list of **variables** involved,
- **precondition** $\to$ a conjunction of literals,
- **effect** $\to$ a conjunction of literals.

A **ground action** is obtained by assigning values to the variables.

## Domain and problem

A set of action schemas defines the **planning domain**. A specific **problem** adds:

- **initial state:** a conjunction of ground fluents,
- **goal state:** a conjunction of literals.

## Applying an action

An action $a$ is applicable in state $s$ **iff** $s\models P$ (its precondition): every positive literal in $P$ must hold in $s$, and every negated literal must **not** be present.

The resulting state:

$$
s'=Result(s,a)=(s-DEL(a))\cup ADD(a)
$$

- start from all the fluents of $s$;
- **remove** the fluents appearing as negated literals in $a$'s effects;
- **add** those appearing as positive literals.

Related: [[Classical Planning]], [[First-Order Logic]].
