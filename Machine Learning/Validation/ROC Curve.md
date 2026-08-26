---
tags: [machine-learning, validation, metrics]
---

# ROC Curve

Plot of **TP rate** (recall/sensitivity) against **FP rate** ($1-$ specificity), obtained by moving the decision threshold of a continuous-output classifier (e.g. the 0.5 of a logistic unit - see [[Activation Functions]]).

- The **diagonal** corresponds to random guessing, i.e. the worst classifier.
- Better curves are above it; the summary number is the **AUC** (Area Under the Curve): higher is better.
- Useful exactly because it does **not** depend on one particular threshold, so it also helps choosing that threshold by studying the effect on FP/FN.

Related: [[Confusion Matrix]], [[Classification Metrics]].
