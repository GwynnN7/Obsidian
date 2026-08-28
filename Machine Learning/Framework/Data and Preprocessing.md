---
tags: [machine-learning, theory]
---

# Data and Preprocessing

## Representation

- **Flat (attribute-value):** fixed-size vectors of features, one table of tuples. This is the default case.
- **Structured:** sequences, trees, graphs, multi-relational data (images, time series, strings, DNA/proteins, molecules, web links). Different models are needed - see [[CNN]] (images), [[RNN]] (sequences) and [[Structured Domain Learning]] (trees, graphs).

Dimensions: $l$ = number of examples, $n$ = number of features. See [[Notation]].

## Encoding categorical variables

- 2 classes: $0/1$ or $-1/+1$.
- Ordered categories: $1,2,3\dots$ (only if the order means something, e.g. small/medium/large - the numbers carry a degree of similarity).
- Symbols with no order: **1-of-k / one-hot**, e.g. $A\to100$, $B\to010$, $C\to001$.

> [!important]
> One-hot encoding on the inputs is a frequent fault. E.g. 6 variables with 2/3/4 symbolic values each give **17 input units**, not 6.

Works for input and output variables alike.

## Scaling

Needed whenever features live on different scales - essential for [[k-NN]] (the metric is dominated by the widest feature), very useful for [[Neural Networks]] (keeps units out of saturation, see [[Weight Initialization]]) and for anything regularized (a penalty on $\lVert w\rVert$ is only fair if the features are comparable).

| | Formula | Output range | Use it when |
|---|---|---|---|
| **Z-score** (standardization) | $\dfrac{v-\mu}{\sigma}$ | unbounded, mean 0, std 1 | the default; features roughly bell-shaped; NN inputs |
| **Min-max** | $\dfrac{v-\min}{\max-\min}$ | $[0,1]$ (or $[a,b]$) | you need a bounded range - image pixels, targets that must sit inside a unit's output range |
| **Max-abs** (absmax) | $\dfrac{v}{\max\lvert v\rvert}$ | $[-1,1]$ | **sparse** data: it does not shift the values, so the zeros stay zeros |

Notes:

- Z-score is unbounded, so it tolerates outliers better than min-max but does not remove them. A robust variant replaces $\mu,\sigma$ with the median and the interquartile range.
- Min-max is the most sensitive to outliers - one extreme value squashes everything else - and new data can fall **outside** $[0,1]$.
- Max-abs is min-max's sparse-friendly cousin: subtracting the min would destroy sparsity by turning every zero into a nonzero.

> [!warning] Fit on the training set only
> Compute $\mu,\sigma$ (or $\min,\max$) on $D_{TR}$ and apply **those same numbers** to $D_{VL}$ and $D_{TS}$. Scaling on the full data set before splitting leaks information from the test set into training - the same mistake as selecting features on everything, see [[Validation Pitfalls]]. Inside a [[Cross Validation]] loop, the scaler is refitted in every fold.

## Dirty data

- **Noise:** external factors added to the signal, not part of the underlying law (e.g. gaussian noise).
- **Outliers:** unusual values not consistent with the rest (abnormal measurements). Either detect and remove them, or use robust methods.
- **Missing data:** careful, $0$ does not mean "no input" if $0$ is a legal value in the input range.
- [[Feature Selection]]: keep the informative variables, drop irrelevant/redundant ones.

Preprocessing is often the most critical phase for the overall success. See [[ML Workflow]].
