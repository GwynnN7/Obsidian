---
tags: [machine-learning, theory]
---

# Cluster Analysis

Unsupervised task: partition the data into clusters (subsets of "similar" data) approximated by a centroid / prototype.

- $H$: a set of vector quantizers $x\to c(x)$, i.e. continuous space $\to$ discrete space.
- Loss: squared error distortion, the distance of a pattern from the centroid of its cluster - see [[Loss Functions]].
- The metric and the representation decide the result.

Also used to pick a subset of prototypes for [[k-NN]].

## Algorithms

- [[Vector Quantization]] - the framework: codebook vectors, Voronoi cells, quantization error.
- [[K-Means]] - the standard algorithm, on-line and batch.
- [[SOM]] - prototypes arranged on a map, so the result can be visualized.

## Evaluating the result

How do you tell a good clustering from a poor one? Mostly you cannot, objectively:

- **Subjective** - gold standards rarely exist outside well-prescribed subdomains.
- **Quantization error** - objective, but it depends on the number of centroids, so it does not compare different $K$ fairly.
- **Purity** (and entropy, mutual information) w.r.t. the labels of an underlying classification task - but if you have the labels you would use a classifier.

Also keep apart the quality of the **model/clustering** and the quality of the **result** in the application, which is domain dependent.

Related: [[Unsupervised Learning]], [[Loss Functions]].
