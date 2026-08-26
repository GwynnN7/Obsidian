---
tags: [machine-learning, validation, metrics]
---

# Confusion Matrix

Table of actual versus predicted classes in a binary classification task.

| | Predicted + | Predicted - |
|---|---:|---:|
| **Actual +** | TP (true positive) | FN (false negative) |
| **Actual -** | FP (false positive, *false alarm*) | TN (true negative) |

All the rates are computed from these four numbers - see [[Classification Metrics]] and [[ROC Curve]].

Generalizes to $K$ classes as a $K\times K$ table, which also shows *which* classes get confused with which.
