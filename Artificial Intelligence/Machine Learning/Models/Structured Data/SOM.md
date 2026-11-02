---
tags: [machine-learning, models]
---

# SOM

Self-Organizing Maps, also called Kohonen maps. An **unsupervised** NN that learns a map from the input space to a lattice of neural units.

## The result

The whole model is two rules, applied to each drawn input $\boldsymbol{x}$ at iteration $t$:

$$
\textbf{1. compete:}\qquad i^*(\boldsymbol{x})=\arg\min_i \lVert \boldsymbol{x}-\boldsymbol{w}_i\rVert
$$

$$
\textbf{2. cooperate:}\qquad \Delta \boldsymbol{w}_i=\eta(t)\ h_{i,i^*(\boldsymbol{x})}(t)\ (\boldsymbol{x}-\boldsymbol{w}_i)
\qquad \forall i
$$

| symbol | what it is |
|---|---|
| $\boldsymbol{w}_i$ | weight (reference/codebook) vector of unit $i$, $\dim(\boldsymbol{w}_i)=\dim(\boldsymbol{x})=n$ |
| $i$ | the unit's **coordinate on the map** (a 2D lattice position, not a data point) |
| $i^*$ | the winner - the unit closest to $\boldsymbol{x}$ |
| $h_{i,j}(t)$ | **neighbourhood function**, decreasing in $\lVert i-j\rVert$; a gaussian of width $\sigma(t)$ |
| $\eta(t)$ | learning rate |

Both $\eta(t)$ and $\sigma(t)$ **decrease with $t$**. Repeat until the weights stop changing.

> [!important] $h$ measures distance on the lattice
> $\lVert i-j\rVert$ is the distance between the **coordinates of the units on the map**, never between their weight vectors. Set $h_{i,j}=\delta_{ij}$ (winner only) and the whole thing collapses to on-line [[K-Means]] - the neighbourhood term *is* the SOM.

## What the pieces mean

- $N$ neurons on a regular **low-dimensional grid** (lattice), usually 2D.
- Each unit is identified by its coordinate on the map.
- Each unit receives the same input $\boldsymbol{x}$ and has its own weight vector.

So the weights are [[Vector Quantization|reference vectors]] like in [[K-Means]] - the new ingredient is that they are **arranged on an ordered map**.

> [!important] Topology-preserving map
> Neighbouring units respond to similar input patterns, and data points close in the input space are mapped onto the same or nearby map units.

## What it is used for

- **Clustering** - the clusters can be read off the map, so it rapidly shows the structure underlying the data.
- **Vector quantization / pattern recognition** - the input is transformed into the weights of the closest unit, i.e. its codebook vector.
- **Data compression** - the input is transformed into the index (digital code) of the winner unit.
- **Projection and exploration** - a multidimensional distribution visualized on a 2D map.

## Competitive learning

An adaptive process in which neurons gradually become **specialized** to different input categories: they compete for an input, and the winner is allowed to learn more.

The biological analogy is the topologically ordered map of the somatosensory (and motor) cortex: adjacent neurons come to represent coincident sources of activation on the body, with a larger area of cortex for the more sensitive parts.

## The two stages in detail

The map is randomly initialized, then for each drawn input $\boldsymbol{x}$:

### 1. Competitive stage

Compare $\boldsymbol{x}$ with all the unit weights, Euclidean distance, winner-take-all. Same rule as [[K-Means]]; and as always the comparison is done on the **squared** distance, since the root cannot change the winner - see [[Vector and Matrix Algebra]].

### 2. Cooperative stage

The winner **and the units in its neighbourhood** are moved closer to the input - Hebbian learning, with $h$ deciding how much each unit gets. In practice the neighbourhood takes a standard geometrical form on the grid (rectangular, hexagonal), so it includes a finite set of map units.

### Decreasing the two parameters

- Wide neighbourhood at the beginning $\to$ **global ordering** of the map.
- Width and learning rate slowly shrinking $\to$ **convergence and local refinement**.

> [!important] Key factor: neighbours on the map
> This is what separates SOM from a plain [[Vector Quantization|soft-max]] adaptation, which uses proximity in the **data** space instead.

## Why the order emerges

The cooperative update is what makes the map topographically ordered: the weights are **not** modified independently, but as topologically related subsets.

- For each subset, similar weight updates are performed.
- At each step the subset is selected by the neighbourhood of the current winner, spatially on the lattice.
- So topological information is fed into the map: the winner and its lattice neighbours receive similar updates, which after training makes them respond to similar inputs.

Natural as it looks, a formal proof of ordering is difficult.

## Visualization

The map can display different features of itself and of the data:

- **Density** of the reference vectors, e.g. colour intensity proportional to the number of hits.
- **U-matrix** - the distance between the prototypes of neighbouring units: dark for a large distance between adjacent units, light for a small one. Clusters appear as light areas separated by dark ridges, with **no a priori assumption on the shape** of the clusters.
- Sammon maps, using lines.

This is what gives SOM its easy interpretability, and it is exactly what [[K-Means]] cannot do.

## Evaluating a clustering

How do you tell a good clustering result from a poor one? See [[Cluster Analysis]] - it is largely **subjective**, gold standards rarely exist, the quantization error depends on the number of centroids, and using the labels of an underlying classification task (purity) is not "the" answer either, otherwise you would use a classifier.

## Other unsupervised topics

Hierarchical clustering (dendrograms), principal components / curves / surfaces, [[Association Rules]], independent component analysis and projection pursuit, multidimensional scaling, t-SNE, generative topographic mapping, ART networks.

## Related

[[Unsupervised Learning]], [[Cluster Analysis]], [[Vector Quantization]], [[K-Means]], [[Neural Networks]], [[RNN]] (SOM extended to sequential and structured domains, see [[Structured Domain Learning]]).
