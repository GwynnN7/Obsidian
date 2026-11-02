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
  i^*(\boldsymbol{x})=\arg\min_p d(\boldsymbol{x},\boldsymbol{x}_p),\qquad d(\boldsymbol{x},\boldsymbol{x}_p)=\sqrt{\sum_{t=1}^{n}(x_t-x_{p,t})^2}=\lVert \boldsymbol{x}-\boldsymbol{x}_p\rVert ,
  $$

  and output $y_{i^*}$. The induced partition of the space is the **Voronoi diagram**.
- **k-NN:** use the neighborhood $N_k(\boldsymbol{x})$ of the $k$ closest patterns and average:

  $$
  \operatorname{avg}_k(\boldsymbol{x})=\frac1k\sum_{\boldsymbol{x}_i\in N_k(\boldsymbol{x})}y_i .
  $$

  - Classification: majority vote,
    $$
    h(\boldsymbol{x})=\arg\max_v \sum_{\boldsymbol{x}_i\in N_k(\boldsymbol{x})}\mathbb{1}_{v,y_i}.
    $$
  - Regression: use the average directly.
- **Weighted variant:** weight the neighbors by $1/d(\boldsymbol{x},\boldsymbol{x}_i)^2$ so that closer ones count more (if $d=0$ for some $i$, return $y_i$).

## Effect of $k$

- Small $k$: very flexible, irregular boundary, 0 training error with $k=1$, high variance $\to$ [[Overfitting]].
- Large $k$: smoother, more robust to noise, higher bias $\to$ [[Underfitting]].
- Effective number of parameters $\approx l/k$, **not** 1. The misclassification-vs-$l/k$ plot is another instance of the SLT U-curve.
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
- [[Curse of Dimensionality]]

## Extensions

Kernel smoothers, local linear regression, prototype methods, case-based reasoning.

The metric idea reappears in [[Kernel Methods]]: a kernel induces a distance $d_k(s,t)=k(s,s)-2k(s,t)+k(t,t)$, and an [[SVM]] with a very narrow RBF kernel behaves like 1-NN.

Related: [[Classification]], [[Regression]], [[Bias-Variance Tradeoff]], [[VC Dimension]] (k-NN has infinite VC-dim).
