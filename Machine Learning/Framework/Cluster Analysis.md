---
tags: [machine-learning, tasks]
---

# Cluster Analysis

Unsupervised task: partition the data into clusters (subsets of "similar" data) approximated by a centroid / prototype.

- $H$: a set of vector quantizers $x\to c(x)$, i.e. continuous space $\to$ discrete space.
- Loss: squared error distortion, the distance of a pattern from the centroid of its cluster - see [[Loss Functions]].
- The metric and the representation decide the result.

Also used to pick a subset of prototypes for [[k-NN]].

Related: [[Unsupervised Learning]], and later [[SOM]].
