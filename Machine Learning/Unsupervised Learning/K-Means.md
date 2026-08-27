---
tags: [machine-learning, models]
---

# K-Means

The simplest and most used algorithm for clustering with a squared-error criterion. It is the direct minimization of the [[Vector Quantization]] loss

$$
E=\sum_i\sum_j \lVert \boldsymbol{x}_i-\boldsymbol{w}_j\rVert^2\ \text{winner}(i,j)
$$

over the reference vectors $\boldsymbol{w}_j$.

## On-line version

Taking the derivatives of $E$ with respect to $\boldsymbol{w}_j$ gives the learning rule of VQ - for each pattern $\boldsymbol{x}_i$:

$$
\Delta \boldsymbol{w}_{i^*}=\eta\ \text{winner}(i,i^*)\,(\boldsymbol{x}_i-\boldsymbol{w}_{i^*})
$$

with $\eta$ the [[Learning Rate|learning rate]]. Only the **winner** prototype moves, and it moves towards $\boldsymbol{x}_i$.

This is stochastic [[Gradient Descent]], so: local minima, and dependence on the initialization.

## Batch version

1. Choose $K$ cluster centers, either coinciding with $K$ randomly chosen patterns, or as $K$ random points inside the hypervolume containing the data.
2. **Assign** each pattern to the closest center (the winner).
3. **Recompute** the centers as the geometric centroid (mean) of their current members.
4. If the convergence criterion is not met, go to 2.

Convergence criteria: no (or minimal) reassignment of patterns, or minimal decrease of the squared error. See [[Stopping Criteria]].

### Details

Given centers $\boldsymbol{w}_1\dots\boldsymbol{w}_K$ (vectors of dimension $n$, like $\boldsymbol{x}$):

**(2)** for each $\boldsymbol{x}$ the winner is the nearest prototype

$$
i^*(\boldsymbol{x})=\arg\min_i\lVert \boldsymbol{x}-\boldsymbol{w}_i\rVert^2,
\qquad
\lVert \boldsymbol{x}-\boldsymbol{w}_i\rVert^2=\sum_{j=1}^{n}(x_j-w_{ij})^2
$$

i.e. the Euclidean distance. Note $\lVert\cdot\rVert^2$ is the **square** of the norm, so a plain sum of squares, no root.

**(3)** for each cluster $i$ the new centroid is the mean of its members

$$
\boldsymbol{w}_i=\frac{1}{|\text{cluster}_i|}\sum_{j\,:\,\boldsymbol{x}_j\in\text{cluster}_i}\boldsymbol{x}_j
$$

a vector, computed from vectors, divided by the number of cluster members.

> [!note] Exercise
> Derive the batch algorithm from the on-line one: sum (average) the on-line $\Delta \boldsymbol{w}$ over the patterns with $\eta=1$, and apply it to obtain $\boldsymbol{w}_{new}=\boldsymbol{w}_{old}+\dots$

## Pros and cons

Popular because it is easy to implement and generally efficient. But:

- **$K$ must be given.** The number of clusters to find is an input, not a result $\to$ [[Hyperparameter Search]].
- **Local minima of $E$**, so the result depends on the initialization and is only a sub-optimal choice of the reference vectors. Standard fix: restart several times, see [[Multiple Minima]].
- It works well for **compact, hyperspherical** clusters - and badly for anything else (elongated, nested or irregular shapes).
- **No visualization**: it does not project the data into a lower-dimensional space.
- The indexing of the $\boldsymbol{w}_i$ is arbitrary, so the mapping is **unordered** - there is no relation between the index of a prototype and its position.

The last two points are what [[SOM]] fixes, by adding a topology over the prototypes.

## Related

[[Vector Quantization]], [[SOM]], [[Cluster Analysis]], [[Unsupervised Learning]], [[k-NN]] (same metric issues, opposite use of the prototypes).
