---
tags: [machine-learning, theory]
---

# Concept Learning

Discrete hypothesis space, learning a Boolean function (a "concept"): e.g. $x$ is a cat if $h_{cat}(x)=1$. This is the setting where the need for an [[Inductive Bias]] is actually **proved**.

## How big is $H$?

With $n$ binary inputs and binary output:

$$
|H| = 2^{\#\text{input instances}} = 2^{2^n}.
$$

For $n=4$ that is $2^{16}=65536$ Boolean functions. After 7 examples $2^{9}$ possibilities are still left.

## Lookup table (rote learner)

Store the examples, answer only if the input matches one of them, otherwise "don't know".

- No inductive bias $\to$ **no generalization**.

## Language bias: conjunctive rules

Restrict $H$ to conjunctions of literals ($l_i$, $\lnot l_i$, don't care), e.g. `if (x2=1) then h(x)=1 else 0`.

Number of semantically distinct hypotheses:

$$
3^n+1
$$

($+1$ because every $h$ containing $l_i \land \lnot l_i$ is equivalent to *false*). For $n=4$: from $65536$ down to $82$.

The [[Version Space]] can then be found efficiently with clever algorithms, without enumerating $H$.

- **But:** if the target concept is not in $H$, it cannot be represented. E.g. `if (x1=1) or (x2=1)` is out of reach.

## Unbiased learner

Take $H=P(X)$, the power set of the instance space (all disjunctions, conjunctions, negations), so that the target is surely inside. With $n=10$ binary inputs, $|X|=1024$ and $|P(X)|=2^{1024}\approx10^{308}$.

> [!important] An unbiased learner cannot generalize
> Every unobserved instance is classified positive by exactly half the hypotheses in the VS and negative by the other half, so the VS gives no answer.
>
> Proof sketch: if $h$ is consistent with the training data, take $h'$ identical to $h$ except $h'(x_i)\neq h(x_i)$ on a test instance $x_i$. They agree on TR, so $h\in VS \Rightarrow h'\in VS$.

Trivial example: $X=\{0,1\}$, $H=\{x,\lnot x,0,1\}$, TR $=\{(0,0)\}$ gives $VS=\{x,0\}$; on the test input $1$ the answer can be either $0$ or $1$. The only way out is to have seen every instance - i.e. the lookup table again.

## Conclusion

- Learning without bias extracts no regularity from the data.
- Restriction/preference bias is needed for **generalization**, not only for efficiency.
- It does not yet tell us *which* bias is best - that question is answered (quantitatively) by [[Statistical Learning Theory]].

Related: [[Hypothesis Space and Inductive Bias]], [[Notation]], and [[Propositional Logic]] for the language these hypotheses are written in.
