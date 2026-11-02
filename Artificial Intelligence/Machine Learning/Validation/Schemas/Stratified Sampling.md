---
tags: [machine-learning, validation]
---

# Stratified Sampling

Stratification = grouping the members of the population into relatively homogeneous subgroups **before** sampling.

For classification: in each random partition (TR/VL/TS in hold-out or CV) every class is represented in approximately the same proportion as in the full data set.

Used to avoid being sensitive to the particular partition (lucky/unlucky sampling), and essential with:

- imbalanced classification data,
- very few data, where a class could disappear from a split entirely.

Also check the **fold composition**: the order in the data may carry a specific meaning.

See [[Cross Validation]].
