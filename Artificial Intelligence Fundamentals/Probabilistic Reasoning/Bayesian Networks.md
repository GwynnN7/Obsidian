---
tags: [artificial-intelligence-fundamental, probability]
---

# Bayesian Networks

A **Directed Acyclic Graph (DAG)** representing the dependencies between random variables.

- **Nodes** = random variables; **directed links** ($A\to B$) typically denote a **causal influence**.
- Each node carries a **Conditional Probability Distribution (CPD)** specifying $P(X\mid Parents(X))$.
- The network structure + the CPDs uniquely define the **full joint distribution**.

## Semantics

**Global:** the full joint is the product of the local conditionals

$$
P(x_1,\dots,x_n)=\prod_i P(x_i\mid Parents(X_i))
$$

e.g. $P(j,m,a,b,e)=P(j\mid a)P(m\mid a)P(a\mid b,e)P(b)P(e)$.

**Local:** each node $X$ is conditionally independent of all its **non-descendants** given its parents.

**Markov blanket:** each node $X$ is conditionally independent of all the other nodes given

$$
MB(X)=Parents(X)\ \cup\ Children(X)\ \cup\ Parents(Children(X)) .
$$

## Building one (causal ordering preferred)

1. Identify the random variables $X_1,\dots,X_n$.
2. Choose an ordering - a **causal** one is generally better.
3. For $i=1..n$: add $X_i$; select the **minimal** set of parents from $X_1,\dots,X_{i-1}$ such that $P(X_i\mid Parents(X_i))=P(X_i\mid X_1,\dots,X_{i-1})$; insert the links and fill the CPT.

This adheres to both semantics and guarantees an acyclic graph, following the simplified chain rule above.

> [!warning]
> The size of a CPT grows **exponentially** with the number of parents ($2^{\#parents}$ for binary variables). Continuous variables would theoretically require CPTs of infinite size.

## Canonical distributions

Ways to avoid filling an exponential table by exploiting known relationships:

- **Deterministic nodes:** $X=f(Parents(X))$ - a compact definition even for continuous variables.
- **Context-Specific Independence (CSI):** $X$ is conditionally independent of *some* of its parents given specific assignments of the others (an if-then-else structure). E.g. in $P(damage\mid accident,use)$, if $\lnot accident$ then $damage$ is CSI of $use$.
- **Noisy-OR:** for multiple **non-interactive binary causes** producing a binary effect. Reduces the CPT from $O(2^{\#parents})$ to $O(\#parents)$.
  - Assumptions: (1) the parent set includes all the relevant causes - otherwise you need a *leak* node; (2) the causal link from any cause can be **independently inhibited**.
  - With $q_i=P(\lnot X\mid U_i,\lnot U_{j\ne i})$ the inhibition probability, the effect is false iff **all** its true parents are inhibited:
    $$P(X\mid U_1,\dots,U_j,\lnot U_{j+1},\dots,\lnot U_k)=1-\prod_{i=1}^{j}q_i$$

## Hybrid networks

Mixtures of discrete and continuous nodes.

- With **parametric** functions (e.g. gaussians) only the parameters need specifying.
- **Discretizing** continuous variables gives large CPTs and approximation errors.
- **Linear Gaussian model:** if all the variables are continuous with linear gaussian distributions (fixed $\sigma$, $\mu$ varying linearly with the parent values), the joint distribution is gaussian.
- With discrete **parents**, the continuous variables are gaussian for any specific assignment of the discrete ones.

Related: [[Bayesian Network Inference]], [[Probability and Decision Theory]], [[Causal Networks]].
