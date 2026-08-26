---
tags: [machine-learning, tasks]
---

# Classification

Predict a discrete class label from input features: $f(x)\in\{1,2,\dots,K\}$.

- Binary: $y\in\{-1,+1\}$ or $\{0,1\}$. With 2 classes it is also called **concept learning** ($f$ is a Boolean function).
- Multiclass: $K>2$, see [[Multiclass Classification]].
- Geometrically it is the allocation of the input space into **decision regions**.

## Loss and metrics

- Inner loss: 0/1 loss, see [[Loss Functions]].
- Metrics: [[Classification Metrics]], [[Confusion Matrix]], [[ROC Curve]].
- Accuracy 50% on a binary task = a coin. Watch out for unbalanced data, where a trivial classifier already looks good.

## Models in these notes

[[Linear Models]] (threshold on $w^Tx$), [[k-NN]] (vote in the neighborhood), [[Neural Networks]] (sigmoidal/softmax output), [[Perceptron]], [[SVM]] (maximum margin).

Related: [[Supervised Learning]], [[Bayes Optimal Classifier]].
