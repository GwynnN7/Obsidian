---
tags: [machine-learning, theory]
---

# Hypothesis Space and Inductive Bias

## Core definitions

- **Model:** describes the relationships in the data through a "language", and so decides which functions can be represented.
- **Hypothesis** $h$: a proposed approximation to the target function $f$.
- **Hypothesis space** $H$: all the hypotheses the learner can output. E.g. $h(x,w)$ with $w$ the parameters.
- **Training example:** $(x, f(x)+\text{noise})$.

## [[Inductive Bias]]

The assumptions that make generalization possible.

- **Language bias:** restrictions on $H$ (which functions we can even write).
- **Search bias:** restrictions/preferences in the [[Learning Algorithm]] that searches $H$.
- Flexible $H$ means low language bias, but then the search is normally heuristic and incomplete - the bias moves into the search.

> [!important]
> An unbiased learner cannot generalize to unseen instances. This is proved in [[Concept Learning]], not just asserted.

In ML we normally prefer **expressive $H$ + search bias** over a restrictive language bias, so that the unknown target is not excluded a priori (Neural Networks, decision trees, ...).

## [[Version Space]]

The subset of $H$ consistent with all the training examples:

$$
VS(H,D)=\{h\in H\mid h(x)=d(x)\ \forall(x,d)\in D\}.
$$

## Examples of inductive bias

| Model | Bias |
|---|---|
| [[Linear Models]] | language: $H$ = linear functions (rigid); search: ordered search driven by least squares |
| [[k-NN]] | the chosen metric defines similarity + locality |
| [[Neural Networks]] | smoothness: small input change $\to$ small output change |
| Conjunctive rules | only AND of literals |

## Related

- [[Generalization and Risk]]
- [[Statistical Learning Theory]]
- [[Overfitting]]
