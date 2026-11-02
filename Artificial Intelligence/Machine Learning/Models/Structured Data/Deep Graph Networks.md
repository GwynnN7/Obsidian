---
tags: [machine-learning, models]
---

# Deep Graph Networks

DGN: the neural framework for learning on graphs. Also met under many other names - neural networks for graphs, graph neural networks (GNN), graph convolutional networks (GCN), graph representation learning, deep learning for graphs.

The domain and the tasks are in [[Structured Domain Learning]]; this note is about how the models work.

## The result

Every DGN is one equation iterated over the graph. For $l=1,\dots,L$ and every node $v$:

$$
\boldsymbol{h}_v^{(l)}=\mathrm{AGG}_{\boldsymbol{W}^{(l)}}\Big(\boldsymbol{L}_v,\ \boldsymbol{h}_v^{(l-1)},\ \{\boldsymbol{h}_u^{(l-1)}:u\in\mathcal{N}(v)\}\Big)
$$

| symbol | what it is |
|---|---|
| $\boldsymbol{h}_v^{(l)}$ | state (embedding) of node $v$ after $l$ steps |
| $\boldsymbol{L}_v$ (or $\boldsymbol{x}_v$) | the label/features attached to $v$ |
| $\mathcal{N}(v)$ | the neighbours of $v$, read off the adjacency matrix $\boldsymbol{A}$ |
| $\mathrm{AGG}$ | a **permutation-invariant** operator over that set (sum, mean, max, attention) plus unit activations |
| $\boldsymbol{W}^{(l)}$ | the free parameters - shared across all nodes |

Three consequences, and they are the whole framework:

1. **Permutation invariance** comes from $\mathrm{AGG}$ being order-blind, and weight sharing across nodes.
2. **After $l$ steps the state of $v$ covers a context of radius $l$** - so depth is not a tuning choice, it is how information reaches distant nodes.
3. **Two ways to iterate:** each step is a new layer with its own weights (*layering / convolutional*: GCN, GraphConv, GAT, NN4G), or the same function is iterated to a fixed point (*recursive*: GNN, GraphESN).

To emit one output for the **whole graph**, aggregate the node states with a permutation-invariant **global pooling** and feed a readout; for node-level tasks the output is emitted per node instead.

## Why not just use a CNN

[[CNN]] and DGN both visit every node of the input, but:

| CNN | Graphs |
|---|---|
| the kernel is applied on a regular 2D grid | the neighbours of a vertex are **unordered** and **variable in size** |
| the neighbours of a pixel are ordered and fixed in number | so a fixed-shape kernel cannot be placed on them |

The fix keeps the two CNN ingredients and drops the grid:

1. **Constrain the receptive field to the neighbours** of the vertex.
2. **Weight sharing** over all the vertices $\to$ **permutation invariance**, i.e. the model abstracts from any ordering of the nodes.
3. Extend beyond the local neighbourhood **through layering**.

## Message passing

Reading the equation at the top:

- Nodes "talk" to each other to inform about themselves and to become aware of the surrounding structure - by **messages**.
- $\boldsymbol{h}_v^{(l)}$ is the **state** (embedding) of node $v$; $\boldsymbol{L}_v$ (or $\boldsymbol{x}_v$) is its label; $\mathcal{N}$ follows the adjacency matrix $\boldsymbol{A}$.
- $\mathrm{AGG}$ contains a **permutation-invariant operator** over the set of neighbours (e.g. a sum) and combinations of unit activations, with $\boldsymbol{W}$ the free parameters. So the node embedding is **learned**.
- Applying it to every node is a **parallel, unordered visit** of the graph at each iteration $l$.

> [!important] Depth is not optional
> One step only sees $\mathcal{N}_v$. Iterating the mechanism spreads/diffuses information from all the nodes to all the others: after $l$ steps the state of $v$ covers a **context of radius $l$**.
>
> Depth is functional to the growth of the context $\to$ **DGNs are inherently deep**.


## Two ways of iterating

### Layering / convolutional

Each message-passing iteration **is a new layer**, with its own weights. For each node the context/receptive field is composed through layers - compositionality, exactly as in [[Deep Learning]].

Some instances (with $\boldsymbol{A}$ the adjacency matrix):

$$
\textbf{GCN:}\quad \boldsymbol{h}^{(l)}=\mathrm{relu}\big(\hat{\boldsymbol{A}}\,\boldsymbol{h}^{(l-1)}\boldsymbol{W}^{(l)}\big),\qquad \hat{\boldsymbol{A}} \text{ degree-normalized}
$$

$$
\textbf{GraphConv:}\quad \boldsymbol{h}^{(l)}=f\big(\boldsymbol{W}^{(l)}_1\boldsymbol{h}^{(l-1)}+\boldsymbol{W}^{(l)}_2\boldsymbol{A}\boldsymbol{h}^{(l-1)}\big)
$$

$$
\textbf{GAT:}\quad \text{attention-based aggregation, with learned edge weights } \alpha_{v,u}
$$

**NN4G** is the constructive version: layers are built **incrementally**, each new layer seeing all the previous (frozen) ones.

$$
h_v^{(1)}=f\Big(\sum_j \bar w_{1j}\,L_j(v)\Big)
$$

$$
h_v^{(l)}=f\Big(\underbrace{\sum_j \bar w_{lj}\,L_j(v)}_{\text{current label of }v}+\underbrace{\sum_{j=1}^{l-1}\hat w_{lj}\sum_{u\in\mathcal{N}(v)}h_u^{(j)}}_{\text{context from the neighbours, from all previous layers}}\Big)
$$

It is **not recursive**: the previous layers are frozen, so there is no gradient propagated through them - no end-to-end backprop through the layers, no vanishing gradient across them, and the number of layers can be decided automatically (greedily expand a layer until the validation accuracy stops improving, then add layers until the same happens). Same spirit as [[Cascade Correlation]].

### Recursive

Here $l$ is the number of message-passing **iterations inside the same layer** (or layers with shared weights). The model is a **dynamical system**: the state iteration is required to converge to a **fixed point**, which becomes the graph embedding, by constraining the state dynamics to be contractive.

- **GNN** - the embedding is trained.
- **GraphESN / GESN** - the message-passing function is **untrained** (randomized), with direct control of the dynamical-system parameters, and only the readout is trained. Reservoir computing extended from sequences ([[Echo State Networks]]) to graphs, with the stability condition now involving the **degree** of the graph, and a deep version stacking reservoir layers $\to$ [[Graph Echo State Networks]].

> [!note] Why bother with untrained message passing
> Beyond efficiency: it **decouples the inductive bias of message passing from the issues of end-to-end training**, which makes it possible to study what the message-passing scheme itself is doing.

## Open issues

| | |
|---|---|
| **Efficiency** | scaling training to very large graphs (>100K nodes) |
| **Expressiveness** | deep models may fail to learn meaningful node representations |
| **Theory** | characterization of the models' inductive bias |

### The over-* issues

- **Over-smoothing** - accuracy typically declines beyond 4-5 layers: node representations **collapse** in the deeper layers, because message passing acts as a **diffusion** and the high-level representations all become similar.
- **Over-squashing** - a bottleneck: the receptive field grows exponentially with depth, but the node embedding has a fixed size. The sensitivity of a node representation to a distant input feature decays exponentially with the number of layers, measured by the Jacobian $\partial\boldsymbol{h}_v^{L}/\partial\boldsymbol{x}_u$. Bounds separate a **Lipschitz** part from a **topology** part:

$$
\textbf{GCN:}\quad \Big\lVert \frac{\partial\boldsymbol{h}_v^{L}}{\partial\boldsymbol{x}_u}\Big\rVert\le\prod_{l=1}^{L}\lVert \boldsymbol{W}^{(l)}\rVert\cdot(\boldsymbol{A}^L)_{u,v}
$$

$$
\textbf{GESN:}\quad \Big\lVert \frac{\partial\boldsymbol{h}_v^{L}}{\partial\boldsymbol{x}_u}\Big\rVert\le\sum_{l=0}^{L}\lVert \boldsymbol{W}\rVert^{l}\,\lVert \boldsymbol{W}^{in}\rVert\cdot(\boldsymbol{A}^L)_{u,v}
$$

so imposing $\lVert \boldsymbol{W}\rVert>1$ on a fixed reservoir prevents the sensitivity from vanishing exponentially.
- For end-to-end training, add the plain difficulty of backpropagating through many message-passing layers.
- How these interact is an open research issue.

**Impact 1 - under-reaching:** the over-* issues can prevent the model from learning interactions between **distant** nodes.

**Impact 2 - heterophily:**

- **Homophilic** graphs: nodes in the same neighbourhood mostly share the same class.
- **Heterophilic** graphs: nodes of the same class are generally further apart, so predictions relying chiefly on the immediate neighbours are misleading.

Measured by the intra-class edge ratio $\dfrac{\text{intra-class edges}}{\text{total edges}}\in[0,1]$: low means heterophilic.

So standard message passing is biased towards **homogeneous** graphs, and that plus the over-* issues makes heterophilic node classification a stress test.

### Dirichlet energy

A way to measure whether a model preserves the variability of the signal across layers:

$$
E(\boldsymbol{h})=\sum_{v\in\mathcal{V}}\ \sum_{u\in\mathcal{N}_v}\lVert \boldsymbol{h}_v-\boldsymbol{h}_u\rVert^2
$$

- **Spatial view:** variability of the embedding between neighbours.
- **Spectral view:** a concise measure of the distribution over the graph frequency spectrum - high-frequency signal $\to$ high energy.
- High spatial variability of the graph signal (high frequency, larger spectra under the graph Fourier transform) goes with **low homophily**, so heterophilic tasks require preserving the high frequencies. Over-smoothing is exactly the loss of that energy through the layers.

### Expressiveness and the WL test

How many graphs can a DGN learn to distinguish?

- The 1-dimensional **Weisfeiler-Leman** heuristic checks a relaxation of graph isomorphism in polynomial time.
- Message passing is similar to WL colouring, so a vanilla DGN is **upper bounded by the WL expressiveness** - it cannot be more discriminating than the test.
- Versions that go beyond 1-WL exist.

## Graph XAI

An explainer assigns an importance score to each node, so it highlights the relevant subgraphs behind a prediction. Used as an **investigation tool** to uncover the inductive bias of the different architectures.

- With the same accuracy, different architectures show **different** detection strategies: some prefer motifs, others local degree patterns.
- The strategy shifts with depth (low-order structures at the lowest layers, motifs higher up) and with the weight values.
- **The truth is not unique**: several models can reach the same accuracy with different explanatory patterns. This matters more for graphs than for images, where a natural visual interpretation exists.
- The practical problem: ground truth is rarely available for real-world graphs, so annotated benchmarks for graph explainability are needed.

## Related

[[Structured Domain Learning]], [[Graph Echo State Networks]], [[CNN]], [[RNN]], [[Echo State Networks]], [[Randomized Neural Networks]], [[Deep Learning]], [[Cascade Correlation]], [[Kernel Methods]].
