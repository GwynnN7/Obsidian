---
tags: [machine-learning, theory]
---

# Structured Domain Learning

Learning when the data are **not vectors**: sequences, trees, graphs, multi-relational data.

> [!important] Why
> Because data have **relationships**, and flattening them throws the relationships away.

Examples of domains that are naturally graphs: molecules (drug design), biological networks (proteins, biochemical pathways), social networks, knowledge graphs, transportation networks, language parse trees, logic terms, scene-understanding networks, brain networks, abstractions of images.

## Where it sits

| Axis | From | To |
|---|---|---|
| Input domain | flat vectors (MLP) | sequences ([[RNN]]) $\to$ trees, graphs |
| Layering | shallow | deep, multi-level input representation |
| Efficiency | fully end-to-end trained NN | randomized / incremental NN |

Deep-and-wide models on structures have a high computational cost, which is why the "deep **and efficient**" direction matters here more than elsewhere - see [[Randomized Neural Networks]] and [[Echo State Networks]].

## The representation problem

There is no systematic way, valid for any task, to extract features or metric relations between structured examples.

- **Feature-based representations** (e.g. topological indexes) are incomplete, or strongly task-dependent.
- **Adjacency / incidence matrices** and other fixed-size representations bring their own problems:
  - over-dimensioned or incomplete - wasteful padding, or loss of information;
  - **alignment** between different graphs;
  - **topological order** - imposing one makes generalization harder.

So this is an instance of [[Representation Learning]] extended to structured domains.

> [!important] Move models to data
> Instead of moving the data to the models (graphs into vectors, trees into sequences, with alignment problems and loss of information), **move the models to the data**.
>
> The ability to treat the proper inherent nature of the input data is the key feature for a successful application.

## The data

**Labeled graphs** $g$:

- a node/vertex $v$ carries a vector label $\boldsymbol{l}(v)$, e.g. $[1,0,1,0.7]$;
- an arc/edge can carry a vector label too (e.g. a position), and can be oriented or not;
- cycles are allowed;
- $\boldsymbol{A}$ is the adjacency matrix, $\mathcal{N}(v)$ the neighbourhood of $v$.

Sequences and trees are special cases, so [[RNN]] and recursive NN are special cases of the same story.

## Transductions

The goal is to learn a mapping between a structured domain and a discrete or continuous space - a **transduction** $T$. Given examples $(\text{graph}_i,\text{target}_i)$, learn $T(\text{graph})$.

| Kind | Output | Example |
|---|---|---|
| **Structure-to-structure** (input-output isomorphic) | one output per node | node classification |
| **Structure-to-scalar/element** | one output per graph | graph classification/regression, e.g. predicting a property or the toxicity of a molecule |
| **Edge-based** | one output per edge | link prediction |
| Non-isomorphic | in general, anything else | |

The molecule case says it well: molecules are not vectors, they are structures of varying size, and we would like to predict the property **directly from the structure** instead of from hand-made descriptors.

## The models

- **Recursive NN for trees** - bottom-up encoding, unfolding through the structure, weight sharing over all vertices and all trees. Already sketched in [[RNN]].
- **Deep Graph Networks** - the general case, based on message passing $\to$ [[Deep Graph Networks]].
- **Kernels for structures** - a different route, below.
- SOM extended to structures, and hidden Markov models for trees, exist too $\to$ [[SOM]], [[Hidden Markov Models]].

## Kernels for structured data

The modularity of [[Kernel Methods]] extends easily to structures: $k(x,x')=\langle\phi(x),\phi(x')\rangle$ works for data of any type, since it only has to define a **similarity** between two objects and embed them implicitly into a Euclidean space.

**Marginalized kernel** (example): similarity between two graphs based on the common label paths obtained by **random walks** on them. The kernel is the inner product of the count vectors, averaged over all the possible label paths. Cost typically scales **quadratically** with the graph size.

**Convolution kernels**: decompose the structures into sub-structures and combine the sub-similarities.

> [!warning] Adaptivity
> A kernel for structures is a **knowledge-driven** method: the similarity measure is fixed **before** learning, so there is no adaptivity in the encoding into the feature space.
>
> - **Pro:** you can inject prior knowledge, engineering the kernel for a specific problem.
> - **Cons:** for structured domains no universality result is known, the choice of the kernel is critical, and computing any *complete* graph kernel is at least as hard as graph isomorphism.
> - Efficiency is critical too, because of the sub-structure decomposition.
>
> Adaptive kernels are a research topic. Compare with the NN route, where the encoding itself is learned - the same "fixed vs learned features" axis as in [[Randomized Neural Networks]].

## Extensions

- **Dynamic / temporal graphs** - relationships or node values that evolve through time (infection spreading, content dissemination on social networks, signals on nodes) $\to$ DynGESN in [[Graph Echo State Networks]].
- Heterogeneous graphs, hypergraphs.
- **Graph transformers** - attention over all the nodes; similar to message passing, many variants, not always better, and with an $O(n^2)$ efficiency issue. See [[Attention and Transformers]].
- Graph generation, foundation models.
- Explainability, and the trustworthy-AI concepts in general - see [[Deep Graph Networks]].

## Applications

Chemistry and cheminformatics (property/activity prediction, drug design and repurposing, toxicology), biology (protein-protein interaction networks, biochemical pathways, predicting dynamical properties without numerical simulation), social networks, NLP.

The rule of thumb: **if your data have relationships - serial, hierarchical, network - do not use a flat representation.**

## Related

[[RNN]], [[Deep Graph Networks]], [[Graph Echo State Networks]], [[Deep Learning]], [[Kernel Methods]], [[Data and Preprocessing]], [[Representation Learning]], [[Echo State Networks]].
