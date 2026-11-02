---
tags: [artificial-intelligence-fundamental, algorithms]
---

# FOL Inference

## Instantiation

- **Universal Instantiation (UI):** replace a universal quantifier with a ground term. Repeated application is **logically equivalent** to the initial KB.
- **Existential Instantiation (EI) / Skolemization:** replace an existential quantifier with a new **Skolem constant or function**. The result is **satisfiability equivalent** but **not logically equivalent** to the original.

## Propositionalization

Repeatedly apply UI in all the possible ways to remove universal quantifiers, converting $KB_{FOL}$ into $KB_{PL}$. Every $KB_{FOL}$ can be propositionalized preserving entailment: for a ground sentence $g$, $KB_{PL}\models g \equiv KB_{FOL}\models g$.

> [!warning] Semidecidability
> Function symbols induce **infinitely many** ground terms. Entailment in FOL is **semidecidable**: if $KB_{FOL}\models\alpha$ a proof exists in a finite $KB_{PL}$; if it is false, the process may never terminate.
>
> Practical approach: **iterative depth-limited search** - build $KB_{PL}$ of depth $n$ for $n=0,1,2,\dots$ and stop when one of them entails $\alpha$.

## Unification

Find a substitution $\theta$ (the **unifier**) making two expressions identical, $\alpha\theta=\beta\theta$. This avoids generating all the possible UI instances - only the ones that actually enable an inference.

$$
Unify(Knows(John,x),\ Knows(y,OJ))=\{x/OJ,\ y/John\}
$$

## Generalized Modus Ponens (GMP)

"Lifts" Modus Ponens from PL to FOL, handling variables through the substitution $\theta$ derived from unification.

- Applies only to **definite clauses** (at most one positive literal).
- **Lifted chaining:** forward and backward chaining work on FOL definite clauses.
  - Forward chaining stays **sound and complete**.
  - Backward chaining, usually implemented as DFS, is **sound but not complete** in infinite search spaces (loops, infinite paths).

## Resolution in FOL

A **complete** inference algorithm for the whole of FOL, going beyond definite clauses - it lifts the resolution rule from PL.

- **Complete:** will find a proof if $KB\models\alpha$.
- **Semidecidable:** may never stop if the statement is not true.

## Conversion to FOL CNF

1. Eliminate biconditionals ($\Leftrightarrow$) and implications ($\Rightarrow$).
   $\forall x\ \lnot(\forall y\ \lnot Animal(y)\vee Loves(x,y))\vee \exists y\ Loves(y,x)$
2. Move negation inwards (De Morgan, $\lnot\forall x\,P\equiv\exists x\,\lnot P$).
   $\forall x\ (\exists y\ Animal(y)\wedge\lnot Loves(x,y))\vee \exists y\ Loves(y,x)$
3. **Standardize variables** so each quantifier uses a unique name.
   $\forall x\ (\exists y\ Animal(y)\wedge\lnot Loves(x,y))\vee \exists z\ Loves(z,x)$
4. **Skolemize:** eliminate $\exists$ with a Skolem function (a constant if the $\exists$ is outside the scope of any $\forall$).
   $\forall x\ (Animal(F(x))\wedge\lnot Loves(x,F(x)))\vee Loves(G(x),x)$
5. **Drop the universal quantifiers** - now implicit.
   $(Animal(F(x))\wedge\lnot Loves(x,F(x)))\vee Loves(G(x),x)$
6. **Distribute $\wedge$ over $\vee$** to reach CNF.
   $(Animal(F(x))\vee Loves(G(x),x))\wedge(\lnot Loves(x,F(x))\vee Loves(G(x),x))$

Related: [[First-Order Logic]], [[Propositional Logic]], [[Logical Inference]].
