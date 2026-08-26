---
tags: [machine-learning, paradigms]
---

# Unsupervised Learning

No teacher: the training set is a set of unlabeled data $\{x\}$.

Typical goals:

- [[Cluster Analysis]] - natural groupings, via [[Vector Quantization]] / [[K-Means]] / [[SOM]].
- Dimensionality reduction, visualization, preprocessing - project multidimensional data into a lower-dimensional space: PCA, multidimensional scaling, independent component analysis, t-SNE.
- Modelling the data density.
- [[Association Rules]].

The practical reason it matters: **unlabeled data are much cheaper to obtain** than labeled ones.

The learned structure depends on the representation, the similarity measure and the assumptions we make.

It also shows up **inside** supervised pipelines: layer-wise pre-training and autoencoders in [[Representation Learning]].

Related: [[Semi-Supervised Learning]], [[SOM]].
