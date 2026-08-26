---
tags: [artificial-intelligence-fundamental, probability]
---

# Causal Networks

Only a **causal ordering** of the variables is permissible - unlike a plain [[Bayesian Networks|Bayesian network]], where any ordering gives a valid factorization.

## Structural equations

The network is described by equations representing **stable mechanisms in nature**:

$$
X_i=f_i(X_{j\ne i})
$$

where $X_j\to X_i$ means $X_j$ is an argument of $f_i$.

## The do-operator (intervention)

$$
do(X=x)
$$

- **Function:** explicitly *assigns* a value to a variable, simulating an action or intervention in the world.
- **Effect:** structurally **removes the variable's probabilistic dependence on its parents** - you cut the incoming arrows.
- **Application:** enables computing **counterfactuals** and predicting the outcomes of interventions.

> [!important]
> This is the formal version of the warning in [[ML Workflow]]: causality cannot be inferred from observational data alone. Seeing $X=x$ and *setting* $X=x$ are different queries, and only the second one predicts what happens if you act.

Related: [[Bayesian Networks]], [[Probability and Decision Theory]].
