---
tags: [machine-learning, models]
---

# k-Nearest Neighbors

Simple but very flexible, and **local**. The other extreme of the panorama w.r.t. [[Linear Models]].

## Eager vs lazy

- **Eager:** analyze the training data and build an explicit hypothesis (linear models, NN).
- **Lazy:** store the training data and wait; when a test point arrives, build an ad hoc hypothesis just for it. This is k-NN.

## Core algorithm

- Store the training data $(x_p,y_p)$, $p=1..l$.
- **1-NN:** find the nearest training example

  $$
  i^*(x)=\arg\min_p d(x,x_p),\qquad d(x,x_p)=\sqrt{\sum_{t=1}^{n}(x_t-x_{p,t})^2}=\lVert x-x_p\rVert ,
  $$

  and output $y_{i^*}$. The induced partition of the space is the **Voronoi diagram**.
- **k-NN:** use the neighborhood $N_k(x)$ of the $k$ closest patterns and average:

  $$
  \operatorname{avg}_k(x)=\frac1k\sum_{x_i\in N_k(x)}y_i .
  $$

  - Classification: majority vote,
    $$
    h(x)=\arg\max_v \sum_{x_i\in N_k(x)}\mathbb{1}_{v,y_i}.
    $$
  - Regression: use the average directly.
- **Weighted variant:** weight the neighbors by $1/d(x,x_i)^2$ so that closer ones count more (if $d=0$ for some $i$, return $y_i$).

## Effect of $k$

- Small $k$: very flexible, irregular boundary, 0 training error with $k=1$, high variance $\to$ [[Overfitting]].
- Large $k$: smoother, more robust to noise, higher bias $\to$ [[Underfitting]].
- Effective number of parameters $\approx l/k$ (Hastie et al.), **not** 1. The misclassification-vs-$l/k$ plot is another instance of the SLT U-curve.
- $k$ is the complexity hyperparameter to select by [[Model Selection and Assessment]] (LOO CV is natural here).

## Inductive bias

- **Distance:** the chosen metric defines similarity. Euclidean by default, but symbolic data need ad hoc metrics (e.g. Hamming distance between strings). Often *this* is the key to a successful application.
- **Locality:** nearby patterns are assumed to have similar targets (local smoothness prior).

## Relation to the [[Bayes Optimal Classifier]]

k-NN directly approximates it: the conditional probability at a point is relaxed to the conditional probability **within a neighborhood**, and the probabilities are estimated by training-sample proportions.

## Practical limits

- **Scale sensitivity:** k-NN is fragile even w.r.t. basic preprocessing. Rescale/standardize the features, otherwise one variable dominates the metric.
- **Cost is deferred to prediction:** distance from the test sample to all stored vectors, so time grows with $l$ (mitigated by indexing / approximate NN search), plus the memory to store all data.
- **No interpretation:** no model is built, no knowledge is extracted; it does not really match the "learning" objective (a learned model can forget the examples).
- Often you must select a subset of prototypes (e.g. by clustering) and a subset of features.

## [[Curse of Dimensionality]]

1. **Hard to find nearby points in high $n$.** Neighborhoods become spatially large and stop being local; shrinking them means lowering $k$ and raising the variance.
   In a unit cube, to capture a fraction $r$ of the volume you need a subcube of side $r^{1/n}$: to capture 10% of the data you need 30% of each coordinate range in 2D, but **80%** in 10D. The other way round: side $0.3$ captures $\sim30\%$ of the volume in 1D, $9\%$ in 2D, $2.7\%$ in 3D and $\sim0.0006\%$ in 10D.
2. **Low sampling density.** Density is proportional to $l^{1/n}$: if 100 points suffice in $\mathbb{R}^1$, you need $100^{10}$ for comparable accuracy in $\mathbb{R}^{10}$.
3. **Irrelevant features.** If the target depends on 2 out of 20 features, the "similarity" is dominated by the 18 irrelevant ones. Fix: feature weighting (stretch the axes) or [[Feature Selection]].

## Extensions

Kernel smoothers, local linear regression, prototype methods, case-based reasoning.

The metric idea reappears in [[Kernel Methods]]: a kernel induces a distance $d_k(s,t)=k(s,s)-2k(s,t)+k(t,t)$, and an [[SVM]] with a very narrow RBF kernel behaves like 1-NN.

Related: [[Classification]], [[Regression]], [[Bias-Variance Tradeoff]], [[VC Dimension]] (k-NN has infinite VC-dim).
