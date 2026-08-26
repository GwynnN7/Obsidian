---
tags: [machine-learning, validation, metrics]
---

# Classification Metrics

Built on the [[Confusion Matrix]]:

| | Predicted + | Predicted - |
|---|---:|---:|
| **Actual +** | TP | FN |
| **Actual -** | FP | TN |

$$
\begin{aligned}
\text{Recall / Sensitivity / TP rate} &= \frac{TP}{TP+FN} \\
\text{Specificity / TN rate} &= \frac{TN}{TN+FP} \\
\text{Precision} &= \frac{TP}{TP+FP} \\
\text{Accuracy} &= \frac{TP+TN}{TP+TN+FP+FN}
\end{aligned}
$$

- **Recall:** share of actual positives correctly identified.
- **Specificity:** share of actual negatives correctly identified. Also $1-$ FP rate.
- **Precision:** share of predicted positives that really are positive.
- **Accuracy:** fraction of correctly classified patterns, often in %. The error rate is $1-$ accuracy.
- FP = false alarm.

> [!warning]
> For a binary task, **50% accuracy is a coin** (random guess predictor). And with unbalanced data (e.g. 99% positives) a trivial classifier already looks excellent - so accuracy alone is not enough.

Threshold-independent view: [[ROC Curve]].

For continuous outputs, see [[Regression Metrics]].

Related: [[Classification]], [[Loss Functions]], [[Validation]].
