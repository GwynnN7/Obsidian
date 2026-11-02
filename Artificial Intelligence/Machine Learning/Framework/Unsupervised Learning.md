---
tags: [machine-learning, moc, theory]
---

# Unsupervised Learning

No teacher: the training set is a set of unlabeled data $\{\boldsymbol{x}\}$.

Almost everything else - [[Theory]], [[Training]], [[Validation]] - is written for the supervised case, where a target tells you what "right" means. Take the target away and most of it stops applying: there is no risk to bound, no error to backpropagate, and no clean way to say a result is good.

The practical reason it matters anyway: **unlabeled data are much cheaper to obtain**.

## Goals

- [[Cluster Analysis]] - find natural groupings.
- **Dimensionality reduction, visualization, preprocessing** - project multidimensional data into a lower-dimensional space: PCA, multidimensional scaling, independent component analysis, t-SNE.
- **Modelling the data density**.
- [[Association Rules]] - which items go together.

## The models

- [[Vector Quantization]] - the framework: codebook vectors, Voronoi cells, quantization error.
- [[K-Means]] - the standard algorithm, on-line and batch.
- [[SOM]] - the same, plus a topology on the prototypes, so the result can be visualized.

## What decides the result

> [!important]
> Without a target, the **metric and the representation are the whole inductive bias.** The distance you assume decides which examples are similar; the algorithm only groups according to it. See the metric-dominance warning in [[Vector Quantization]] - the same one as in [[k-NN]].

And evaluating the outcome is largely subjective - see [[Cluster Analysis]].

## Where it shows up in supervised work

- Layer-wise pre-training and autoencoders $\to$ [[Representation Learning]].
- Picking a subset of prototypes for [[k-NN]].
- [[Semi-Supervised Learning]] - a mix of the two paradigms.

Related: [[ML Framework]], [[Machine Learning]].
