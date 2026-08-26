---
tags: [machine-learning, ensembles]
---

# Boosting

(e.g. AdaBoost.) Train weak learners **sequentially**, each one concentrating on what the previous ones got wrong, then combine them with weighted votes.

1. Train the 1st classifier.
2. Train the 2nd giving more weight to the instances misclassified before (they are more likely to end up in its training set).
3. And so on.
4. Combine by output weights: more weight to the classifiers with lower error.

## Properties

- A **weak learner** only needs to do better than random guessing (error $<1/2$ for binary classification).
- If not stopped, boosting learns to classify all the training instances correctly - an incremental way to build complex models.
- Can greatly improve weak learners.
- Reduces **bias**, and the variance does not increase in exchange (it resists overfitting).
- Like [[Bagging]], tends to produce separation curves "in the middle" (max margin).
- **But:** sensitive to noisy data, since noisy/hard examples get weighted more. It does not solve the under/overfitting issue by itself.

Related: [[Ensemble Learning]], [[Bias-Variance Tradeoff]].
