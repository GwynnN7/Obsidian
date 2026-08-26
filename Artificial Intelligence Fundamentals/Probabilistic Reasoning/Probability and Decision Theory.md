---
tags: [artificial-intelligence-fundamental, probability]
---

# Probability and Decision Theory

## Why uncertainty is unavoidable

Three reasons, in large domains:

- **Laziness:** compiling a complete, guaranteed set of deterministic rules is prohibitive.
- **Theoretical ignorance:** complete theoretical frameworks are frequently unavailable (medical science).
- **Practical ignorance:** even with all the rules known, unforeseen external factors make them fail.

Rational decisions therefore need both the **importance** of goals (utility) and the **likelihood** of achieving them (probability).

## The three theories

| | Function |
|---|---|
| **Probability theory** | quantify and summarize uncertainty by assigning a degree of belief to propositions |
| **Utility theory** | quantify the quality or intrinsic usefulness of an outcome, so preferences among outcomes can be expressed |
| **Decision theory** | preferences + probabilities |

- **Bayesian view:** start from an initial belief (**prior**), update it to a refined belief (**posterior**) using observed data (**likelihood**).
- **Frequentist view:** assume an underlying true probability, estimated by the frequency of events over repeated trials.

> [!important] Maximum Expected Utility (MEU)
> A rational agent selects the action maximizing the **expected utility**, i.e. the weighted average over all the possible outcomes of that action. This requires a continuous degree of belief.

See [[Rational Agents]], [[Stochastic Games]].

## Core probability

- **Conditional probability:** $P(a\mid b)=\dfrac{P(a\wedge b)}{P(b)}$
- **Product rule:** $P(a\wedge b)=P(a\mid b)P(b)$
- **Independence:** $P(a\mid b)=P(a)$, hence $P(a\wedge b)=P(a)P(b)$
- **Chain rule:** repeated application of the product rule

**Representation:** variables are **random variables**; a **probability distribution** maps every value of a random variable to its probability.

## Inference with the joint distribution

- **Joint probability** $P(a,b)$ - the full joint distribution can be written as a table.
- **Marginalization:** $P(A)=\sum_b P(A,B=b)$.
- **Goal:** the posterior of the query variables $Y$ given the observed evidence $E=e$. With $H=X-Y-E$ the hidden variables:
  $$P(Y\mid E=e)=\alpha\,P(Y,E=e)=\alpha\sum_h P(Y,E=e,H=h)$$
  with $\alpha$ a normalizing constant.
- **Complexity:** $n$ variables of $d$ values each give $d^n$ entries $\to$ $O(d^n)$ time and space in the worst case.

**Independence is what saves us:** if all $n$ variables are independent, the joint needs only $d\cdot n$ entries instead of $d^n$.

## Bayes' rule

$$
P(b\mid a)=\frac{P(a\mid b)P(b)}{P(a)}
$$

A prior $P(b)$ updated to a posterior $P(b\mid a)$ after observing $a$, using the likelihood $P(a\mid b)$.

$$
P(cause\mid effect)=\frac{P(effect\mid cause)P(cause)}{P(effect)}
$$

- $P(effect\mid cause)$ is the **causal** direction,
- $P(cause\mid effect)$ is the **diagnostic** direction.

Same rule behind the [[Bayes Optimal Classifier]].

## Conditional independence

$A$ and $B$ are conditionally independent given $C$ if

$$
P(A,B\mid C)=P(A\mid C)P(B\mid C) \iff P(A\mid B,C)=P(A\mid C)
$$

Applied to multiple effects of a single cause:

$$
P(effect_1,\dots,effect_n\mid cause)=\prod_j P(effect_j\mid cause)
$$

This is the property [[Bayesian Networks]] are built on - and, with a single cause node, it is exactly Naive Bayes.

Related: [[Bayesian Networks]], [[Social Welfare]].
