---
tags: [artificial-intelligence-fundamental, algorithms, models]
---

# NEAT

*NeuroEvolution through Augmenting Topologies* - evolve both the **topology** and the **weights** of a [[Neural Networks|neural network]], starting from minimal solutions.

## CPPN - Compositional Pattern Producing Networks

A popular technique for implementing **indirect encodings** (see [[Genetic Algorithms]]).

- The network maps **phenotypical space coordinates** (input) to a **phenotypical value** (output). The function itself is the encoding, across all input coordinates.
- Connections and nodes are evolved, e.g. by a GA whose genes encode network elements.
- **Complexification:** networks start small and grow, supporting hierarchical development.

## NEAT

- **Encoding:** direct encoding of all the nodes and connections in the genome.
- **Mutation:**
  - random perturbation of existing connections,
  - creation of a new randomly-weighted connection,
  - addition of a new node by **splitting** an existing connection (the original is disabled).
- **Crossover:** uses an **innovation number** to globally track the origin of structural mutations, so that crossover stays coherent.
  - **Matching genes** are inherited randomly.
  - **Disjoint/excess genes** are inherited from the fitter parent (or discarded).
- **Speciation:** maintains diversity and protects new, low-performing networks by restricting competition to similar niches.
  - Innovation numbers give a distance in topology space:
    $$\delta=c_1\frac{E}{N}+c_2\frac{D}{N}+c_3\overline{\Delta W},\qquad \delta_t=\text{compatibility threshold}$$
    with $E,D$ = excess/disjoint genes, $N$ = number of genes, $\overline{\Delta W}$ = average weight difference of the matching genes.
  - Fitness sharing: for individual $i$ of species $s$, $f_i^s=f_i/\lvert s\rvert$.

## HyperNEAT

CPPNs evolved **using NEAT** to generate the connectivity pattern of a **substrate** network (indirect encoding).

- The CPPN output determines the value of a connection from a 4-dimensional point in hyper-cube space (the coordinates of both endpoints).
- Benefit: modularity **at the connection level**.

> [!note]
> This is architecture search by evolution. Compare with the constructive approach of [[Cascade Correlation]], which also grows the network but greedily and by gradient - and with [[Randomized Neural Networks]], which give up on adapting the hidden layer altogether.

Related: [[Genetic Algorithms]], [[Evolutionary Algorithms]], [[Neural Networks]].
