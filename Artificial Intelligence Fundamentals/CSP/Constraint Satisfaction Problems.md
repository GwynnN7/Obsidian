---
tags: [artificial-intelligence-fundamental, csp]
---

# Constraint Satisfaction Problems

**Factored representation:** domain knowledge is formally defined by a set of constraints $C$.

**Goal:** assign values to a set of variables such that all the constraints in $C$ are simultaneously satisfied.

## Terminology

- **Complete assignment:** all variables have a value.
- **Partial assignment:** only a subset does.
- **Consistent assignment:** partial or complete, it satisfies all the relevant constraints.

## Constraint graph

Variables are nodes, constraints are arcs connecting the constrained variables. The **structure** of this graph decides how hard the problem is - see [[Tree-Structured CSPs]].

## Why not plain search

States are defined by the current set of assigned values. A complete solution sits at depth $n$, so a naive search space is $O(n!\,d^n)$ with $n$ variables and domain size $d$. [[Backtracking Search]] cuts this to $O(d^n)$ leaves.

Related: [[Backtracking Search]], [[Constraint Propagation]], [[Search Problems]].
