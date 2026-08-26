---
tags: [machine-learning, models, neural-networks]
---

# SOM

Self-Organizing Maps, also called Kohonen maps. An **unsupervised** NN that learns a map from the input space to a lattice of neural units.

- $N$ neurons on a regular **low-dimensional grid** (lattice), usually 2D.
- Each unit is identified by its **coordinate on the map**.
- Each unit receives the same input $\boldsymbol{x}$ and has a weight vector $\boldsymbol{w}$, with $\dim(\boldsymbol{w})=\dim(\boldsymbol{x})=n$.

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

## The algorithm

The map is randomly initialized, then for each drawn input $\boldsymbol{x}$:

### 1. Competitive stage

Compare $\boldsymbol{x}$ with all the unit weights, Euclidean distance, winner-take-all:

$$
i^*(\boldsymbol{x})=\arg\min_i \lVert \boldsymbol{x}-\boldsymbol{w}_i\rVert
$$

where $i$ is an index **on the map** (e.g. a 2D coordinate). Same rule as [[K-Means]].

### 2. Cooperative stage

The winner **and the units in its neighbourhood** are moved closer to the input (Hebbian learning). At iteration $t$:

$$
\Delta \boldsymbol{w}_i=\eta(t)\ h_{i,i^*(\boldsymbol{x})}(t)\ (\boldsymbol{x}-\boldsymbol{w}_i)
$$

with $h_{i,j}(t)$ the **neighbourhood function** (or kernel), a function that decreases monotonically as $\lVert i-j\rVert$ grows - the distance between the **coordinates** of the units on the lattice, not between their weights. A gaussian of width $\sigma$ is the usual choice.

Repeat until convergence (no more changes).

### Decreasing the two parameters

Both the learning rate $\eta$ and the neighbourhood radius $\sigma$ **decrease with $t$**:

- wide neighbourhood at the beginning $\to$ global ordering of the map;
- width and height slowly shrinking during learning $\to$ convergence and local refinement.

In practice the shape of the neighbourhood is taken from a standard geometrical form on the grid (rectangular, hexagonal) so it includes a finite set of map units.

> [!important] Key factor: neighbours on the map
> This is what separates SOM from a plain [[Vector Quantization|soft-max]] adaptation, which uses proximity in the **data** space instead.

## Why the order emerges

The cooperative update is what makes the map topographically ordered: the weights are **not** modified independently, but as topologically related subsets.

- For each subset, similar weight updates are performed.
- At each step the subset is selected by the neighbourhood of the current winner, spatially on the lattice.
- So topological information is fed into the map: the winner and its lattice neighbours receive similar updates, which after training makes them respond to similar inputs.

Natural as it looks, a formal proof of ordering is difficult.

> [!note] Exercise
> Focus on the differences between K-means and SOM: winner-only vs winner-and-neighbours, arbitrary indexing vs ordered lattice, no projection vs 2D visualization.

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
