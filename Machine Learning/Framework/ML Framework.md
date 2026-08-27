---
tags: [machine-learning, moc]
---

# ML Framework

General frame is **learn a function from examples**. The five topis of any ML system:

| Ingredient | What it is | Note |
|---|---|---|
| **Data** | the available experience | [[Data and Preprocessing]] |
| **Task** | what we want out of it | [[Classification]], [[Regression]], [[Cluster Analysis]], [[Association Rules]] |
| **Model** | the "language" of the hypotheses, defines $H$ | [[Models]] |
| **Learning algorithm** | the search inside $H$ | [[Learning Algorithm]] |
| **Validation** | does it generalize? | [[Validation]] |

## Learning as function approximation

- Given: training examples $(x, f(x)+\text{noise})$ for an unknown $f$.
- Find: a hypothesis $h$ that approximates $f$ well **on new data**, not just on the training set.
- The target $d$ (or $y$, or $t$) is what the teacher gives us.
- Same frame for classification (discrete output) and regression (continuous output).

## Paradigms

- [[Supervised Learning]] - labeled examples $(x,d)$.
- [[Unsupervised Learning]] - no labels, find structure. It has its own folder, since most of the supervised machinery does not apply to it.
- [[Semi-Supervised Learning]] - a mix.
- [[Reinforcement Learning]] - learn from reward, not from step-by-step targets.

## When is ML the right tool

Useful when:

- there is no good theory / no rules we can write by hand,
- data are noisy, uncertain or incomplete,
- the behaviour must be personalized.

Requires:

- a representative source of training data,
- some tolerance on the precision of the answer.

> [!warning]
> Inferring a function from finite data is an **ill-posed (inverse) problem**: existence, uniqueness or stability of the solution can all fail. This is exactly why we need an [[Inductive Bias]].

## Related

- [[Loss Functions]] - how we measure the quality of the approximation.
- [[Notation]] - symbols used everywhere.
- [[ML Workflow]] - the practical design cycle.
- [[Generalization and Risk]] - the point of the whole thing.
