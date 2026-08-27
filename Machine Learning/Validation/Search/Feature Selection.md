---
tags: [machine-learning, validation, preprocessing]
---

# Feature Selection

Find the subset of features that are most informative for the problem at hand, and drop irrelevant/redundant ones.

## Benefits

- **Dimensionality reduction** - VC-dim effects for NN, [[Curse of Dimensionality]] for k-NN - and filtering of irrelevant information and noise. Can give a large performance improvement.
- **Interpretability** - you end up with a set of useful descriptors.

## Cost

- Computationally **hard**: think of how many subsets of features exist, and that each one requires retraining the model.
- Typically a heuristic search of the best subset (including or not the learner) - greedy, GA, or other optimization techniques.

## The trap

> [!warning] Feature subset selection bias
> The results are particularly sensitive to the validation method. Selecting the features on the **entire data set** is wrong: the test set has then been implicitly used, and the estimate is optimistic.
>
> Feature selection must happen **inside** the model-selection process, i.e. inside each fold. See [[Validation Pitfalls]] for the counterexample where this turns a 50% (random) task into an apparent 97%.

Related: [[Data and Preprocessing]], [[Model Selection and Assessment]], [[Regularization]] (L1 does an implicit selection).
