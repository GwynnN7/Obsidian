---
tags:
  - machine-learning
  - subject
  - moc
---

# Machine Learning

Machine Learning hub.

## Three ways in

| | | |
|---|---|---|
| **See it** | [[ML Map]] | the canvas: the whole path as boxes and arrows |
| **Read it** | [[ML Essentials]] | one page: the concepts that matter, in order, with the rules |
| **Browse it** | the table below | the shelves, when you already know what you are looking for |

## The shelves

| Folder | What is in it | Inside |
|---|---|---|
| [[ML Framework\|Framework]] | the setup | data, tasks, paradigms, loss, notation, workflow |
| [[Theory]] | why learning works | `Inductive Bias`, `Generalization`, `Capacity and Bounds` |
| [[Models]] | what can be represented | `Linear and Kernel`, `Instance-Based`, `Neural Networks`, `Architectures`, `Deep Learning`, `Structured Data`, `Ensembles` |
| [[Training]] | how the weights are found | `Algorithm`, `Knobs`, `Complexity Control` |
| [[Validation]] | how you know it works | `Schemas`, `Metrics`, `Search` |
| [[Unsupervised Learning]] | the branch with no teacher | clustering, vector quantization, maps |
| [[Hyperparameter Guidelines]] | the practical do / don't reference | - |

> [!note] Folders are just shelves
> Links do not care where a note lives, so the folders are only there to make browsing sane. The **order** of the subject lives in [[ML Essentials]] and [[ML Map]], not in the tree.

## The three things to never forget

> [!important] Gold rule
> $D_{TR}$ trains; $D_{VL}$ selects; $D_{TS}$ estimates the final risk. Once a set influences a design choice, it is no longer an unbiased test set. See [[Validation Pitfalls]].

> [!important] The trade-off
> Fitting capability *vs* model complexity. Same idea in two languages: $R\le R_{emp}+\epsilon(1/l,VC,1/\delta)$ ([[Statistical Learning Theory]]) and $\text{bias}^2+\text{variance}+\text{noise}$ ([[Bias-Variance Tradeoff]]).

> [!important] Bias is necessary
> A learner with no [[Inductive Bias]] is a lookup table: it cannot generalize. Proved in [[Concept Learning]].

Part of [[Artificial Intelligence]].
