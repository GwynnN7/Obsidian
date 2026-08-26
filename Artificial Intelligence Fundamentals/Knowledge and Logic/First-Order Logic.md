---
tags: [artificial-intelligence-fundamental, logic]
---

# First-Order Logic

More expressive than [[Propositional Logic]]: it can reason about

- **Objects** (person, number, colour),
- **Relations / properties** - $red(x)$, $brotherOf(x,y)$,
- **Functions** - properties with exactly one value for each input.

## Sentence structure

- **Term $t$:** a logical expression referring to an object - a constant, a variable, or a function applied to terms $f(t_1,\dots,t_n)$.
- **Atomic sentence:** a predicate applied to terms $p(t_1,\dots,t_n)$, or an equality $t_1=t_2$.
- **Complex sentence:** atomic sentences joined by connectives.

## Truth value

Evaluated w.r.t. a **model** and an **interpretation**:

- **Model:** the set of domain elements and their relations.
- **Interpretation:** associates a referent (constant, predicate, function) to each symbol.

## Quantification

| | Reads | Pairs with | Example |
|---|---|---|---|
| **$\forall$** universal | true if $P$ is true for **every** element - a possibly infinite **conjunction** | **implication** $\to$ | $\forall x\ Attend(x,AIF)\to Smart(x)$ |
| **$\exists$** existential | true if $P$ is true for **at least one** element - a possibly infinite **disjunction** | **conjunction** $\wedge$ | $\exists x\ Attend(x,GDL)\wedge Bored(x)$ |

## Substitution

$$
S=Smarter(x,y),\qquad \sigma=\{x/Cossu,\ y/Bacciu\},\qquad S\sigma=Smarter(Cossu,Bacciu)
$$

## Building a FOL KB

1. Identify the questions.
2. Assemble the knowledge.
3. Decide the vocabulary.
4. Encode the domain knowledge.
5. Encode the description of the problem instance.
6. Ask queries to the inference algorithm.
7. Debug and evaluate the knowledge base.

Related: [[FOL Inference]], [[Knowledge-Based Agents]].
