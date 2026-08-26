---
tags: [machine-learning, validation]
---

# Cross Validation

The term is used generically for two things:

- an approach to **model selection** based on direct estimation of the prediction error by resampling (instead of analytically);
- more specifically, the implementations that say **how to partition the data**, for both selection and assessment (e.g. K-fold CV).

## K-fold CV

1. Split the data set $D$ into $k$ mutually exclusive subsets $D_1,\dots,D_K$.
2. Train the learning algorithm on $D\setminus D_i$.
3. Test it on $D_i$.
4. Repeat for all the folds; aggregate.

- It uses all the data for training and for testing.
- You also get a **variance / standard deviation** over the folds for your class of models - report `MEAN +/- SD`.
- It can be used for the validation set **or** for the test set (or both, nested).

> [!important]
> Used for **model selection**, K-fold CV does return a final model: after picking the best $\Lambda$, you **retrain on the whole data set** (see [[Validation Schemas]]).
> Used for **assessment** (or in double CV), it does *not* give a unique model - you get one per fold, and what you take away is a risk estimate with its variability.

## Variants

- **Holdout:** cheap, but split-dependent / unreliable.
- **Leave-One-Out** ($K=l$): almost no data wasted, but expensive and potentially high variance.
- **5-fold or 10-fold:** recommended as a good compromise between bias and variance of the estimate w.r.t. LOO (Hastie et al.).
- **Repeated holdout / repeated CV:** repeat the split with different random sampling (e.g. repeat the CV 10 times), average the results. Especially useful for **comparing** models.
- [[Stratified Sampling]]: preserve class/group proportions.
- [[Bootstrap]]: resample with replacement.

## Nested / double CV

- **Inner CV:** model/hyperparameter selection.
- **Outer fold/CV:** assessment of the whole selection procedure.

Written out in [[Validation Schemas]] (5d).

## Lucky / unlucky sampling

Can we avoid being sensitive to the particular partition (the bias of the particular sample)?

- **Stratification:** group the members of the population into relatively homogeneous subgroups before sampling. For classification, each class is represented in approximately the same proportion in every partition as in the full data set.
- **Fold composition:** check whether the order in the data has a specific meaning.
- **Repeated holdout / CV:** average over several random splits.

## Very few data

Hard to say whether a sample is representative. Watch out for (or at least account for):

- missing classes or features in the training data;
- special classes of data not sampled in TR or TS (again, stratification);
- known outliers, which can affect the mean test results;
- the opposite: selecting only "easy" cases.

A blind test set can be misleading too, if it comes from a different distribution, was measured with a different scale/tolerance, is uncleaned/unpreprocessed, or requires extrapolation out of range.

## Choosing a model with CV: how to read the result

Whichever model class gives the best CV score, train it with all the data - that is your predictive model.

> [!warning]
> The best is chosen by looking at the **validation** result, NOT by considering or averaging in the TR error.

Typical examples: comparing different algorithms; choosing #hidden units in a 1-hidden-layer NN (10, 20, 30, ...); choosing $k$ for [[k-NN]] with LOOCV.

## Related

- [[Model Selection and Assessment]], [[Validation Schemas]], [[Validation Pitfalls]]
- [[Early Stopping]] inside CV
- [[Hyperparameter Search]]
