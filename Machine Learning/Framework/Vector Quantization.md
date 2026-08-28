---
tags: [machine-learning, theory]
---

# Vector Quantization

The framework used here for **partitioning clustering**: encode a data manifold with a finite set of reference vectors, and describe each pattern by the closest one.

Given a submanifold $V\subseteq\mathbb{R}^n$, use only a finite **codebook** $\boldsymbol{w}=(\boldsymbol{w}_1,\dots,\boldsymbol{w}_K)$, with $\boldsymbol{w}_i\in\mathbb{R}^n$.

A data vector $\boldsymbol{x}\in V$ is described by its **winner** (best-matching) reference vector $\boldsymbol{w}_{i^*(\boldsymbol{x})}$, the one for which the distortion $d(\boldsymbol{x},\boldsymbol{w}_{i^*(\boldsymbol{x})})$ is minimal.

> [!note] Many names for the same thing
> Reference vector, centroid, cluster center, codebook vector, prototype vector - they all denote $\boldsymbol{w}_i$.

## Voronoi polyhedra

The codebook divides $V$ into subregions:

$$
V_i=\{\boldsymbol{x}\in V \ : \ \lVert \boldsymbol{x}-\boldsymbol{w}_i\rVert\le\lVert \boldsymbol{x}-\boldsymbol{w}_j\rVert \ \ \forall j\}
$$

Each cell contains all the points closer to $\boldsymbol{w}_i$ than to any other reference vector; the segments of the diagram are the points equidistant from two of them.

Our problem is the reverse of drawing the diagram: **given the data, find the centers that minimize the loss.**

> [!warning]
> Computing the optimal vector quantization codebook is **NP-complete**.

## The loss: quantization error

With the squared error criterion $d(\boldsymbol{x}_i,c(\boldsymbol{x}_i))=\lVert \boldsymbol{x}_i-c(\boldsymbol{x}_i)\rVert^2$, the loss is the average distortion over the input distribution, also called **reconstruction** or **quantization error**:

$$
E=\int f\big(d(\boldsymbol{x},\boldsymbol{w}_{i^*(\boldsymbol{x})})\big)\,p(\boldsymbol{x})\,d\boldsymbol{x}
 =\int \lVert \boldsymbol{x}-\boldsymbol{w}_{i^*(\boldsymbol{x})}\rVert^2\,p(\boldsymbol{x})\,d\boldsymbol{x}
$$

with $p(\boldsymbol{x})$ the probability distribution of the data. The discrete version over the data set:

$$
E=\sum_i\sum_j \lVert \boldsymbol{x}_i-\boldsymbol{w}_j\rVert^2\ \text{winner}(i,j)
$$

where $\text{winner}(i,j)$ is the characteristic function of the receptive field of $\boldsymbol{w}_j$: $1$ if $j$ is the winner for $\boldsymbol{x}_i$, $0$ otherwise.

This is the clustering loss already listed in [[Loss Functions]].

> [!note] Differentiability
> The integrand is not continuously differentiable, because $i^*(\boldsymbol{x})$ changes discretely with $\boldsymbol{x}$. We take the derivatives **locally**, fixing $\boldsymbol{x}$ inside one Voronoi cell - which is exactly what gives the [[K-Means]] update rule.

Minimizing $E$ in $\boldsymbol{w}$ **is** the VQ problem.

## VQ and clustering

They are not the same thing:

- Clustering has the more general goal of finding *interesting/useful* groupings, and "interesting" is often defined implicitly by the algorithm, not necessarily by a minimum quantization error.
- But VQ is a clean framework, and VQ algorithms are commonly used for clustering.

## The metric decides everything

> [!important] Metric dominance
> The assumed distance decides which examples are similar; the algorithm only groups according to that metric.

The Euclidean distance is popular but it is **just a choice** - the same warning as in [[k-NN]]. Choosing another metric can have deeper implications, e.g. weighting outliers softly with $1-$ a gaussian centered on the prototype.

## Soft-max adaptation

A common way to avoid getting stuck in local minima: do **not** modify only the winner.

The adaptation to $\boldsymbol{x}$ affects all the centers depending on their proximity to $\boldsymbol{x}$, typically with a step size that decreases with $d(\boldsymbol{x},\boldsymbol{w}_i)$ - e.g. maximum-entropy clustering with gaussian distances.

The NN instance of this strategy is [[SOM]], where proximity is measured **on the map** instead of in the data space.

## One dimension

Quantization in 1D is the digital-signal-processing operation: approximate a continuous range of values with a small finite set of discrete symbols, accepting a quantization/distortion error. Music on a digital device is exactly this.

## Related

[[Cluster Analysis]], [[K-Means]], [[SOM]], [[Unsupervised Learning]], [[Loss Functions]], [[k-NN]].
