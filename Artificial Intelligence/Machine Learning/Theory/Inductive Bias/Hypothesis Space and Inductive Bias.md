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
> An unbiased learner cannot generalize to unseen instances. This is proved in [[Concept Learning]].

In ML we normally prefer **expressive $H$ + search bias** over a restrictive language bias, so that the unknown target is not excluded a priori.

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

## The equivalent deductive system

A useful way to see what a bias *is*: an inductive learner with a bias is **equivalent to a deductive system**.

| | input | machinery | output |
|---|---|---|---|
| Inductive | training examples + new instance | learning algorithm over $H$ | classification of the new instance, or *don't know* |
| Deductive | training examples + new instance + **the inductive bias as an explicit assertion** | theorem prover | the same classification, or *don't know* |

So the bias is exactly the set of extra premises you would have to write down to turn the induction into a deduction. Two consequences:

- a learner with **no** bias has nothing to add to the premises, so it can prove nothing about unseen instances - the lookup table again;
- comparing two learners means comparing the assertions their biases correspond to.

## Language bias or search bias?

In practice ML tends to prefer the **search** bias:

- use an expressive $H$ (neural networks, decision trees - models with universal capability), so the unknown target is not excluded a priori by the language;
- keep a bias, but put it in the **search**, where the learning algorithm rules it;
- concretely, this means using an *incomplete* search strategy over a big space rather than a complete search over a small one.

> [!important]
> Learning without bias extracts no regularity - it is a lookup table with no generalization. Every real ML approach shows an inductive bias; the useful work is characterizing *which* bias each model and each learning algorithm has.

## Related

- [[Generalization and Risk]]
- [[Statistical Learning Theory]]
- [[Overfitting]]
