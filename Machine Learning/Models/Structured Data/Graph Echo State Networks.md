---
tags: [machine-learning, models]
---

# Graph Echo State Networks

GESN / GraphESN: **reservoir computing on graphs**. The message-passing function is randomized and left **untrained**, and only the readout is learned.

It is the third corner of the same idea:

| Domain | Model |
|---|---|
| vectors | random-weights NN $\to$ [[Randomized Neural Networks]] |
| sequences | ESN $\to$ [[Echo State Networks]] |
| graphs | **GESN** - this note |

The general framework it belongs to is in [[Deep Graph Networks]]; here we look at what changes when nothing in the encoder is trained.

## The result

Three equations and one condition.

**Encoder** (random, untrained), for every vertex $v$:

$$
\boldsymbol{x}(v)=\tanh\Big(\boldsymbol{W}^{in}\boldsymbol{u}(v)+\sum_{v'\in\mathcal{N}(v)}\hat{\boldsymbol{W}}\,\boldsymbol{x}(v')\Big)
$$

**Stability** - with $k$ the maximum degree of the graphs in the data:

$$
\textbf{sufficient:}\quad \lVert \hat{\boldsymbol{W}}\rVert\cdot k<1
\qquad\qquad
\textbf{necessary:}\quad \rho(\hat{\boldsymbol{W}})\cdot k<1
$$

**Readout** (the only trained part), linear, in closed form:

$$
y(\mathcal{G})=\boldsymbol{W}^{out}\sum_{v\in\mathcal{V}}\boldsymbol{x}(v)
$$

> [!important] The one thing to remember
> In a plain [[Echo State Networks|ESN]] the stability condition is on $\rho(\hat{\boldsymbol{W}})$ alone. On graphs the **topology joins the condition** through $k$: the more connected the graph, the more signal each node receives, so the weights must be scaled down to compensate.

## The encoder

Each vertex $v$ gets a state $\boldsymbol{x}(v)$, computed from its label $\boldsymbol{u}(v)$ and the states of its neighbours:

$$
\boldsymbol{x}(v)=\tanh\Big(\boldsymbol{W}^{in}\boldsymbol{u}(v)+\sum_{v'\in\mathcal{N}(v)}\hat{\boldsymbol{W}}\,\boldsymbol{x}(v')\Big)
$$

or, for all the vertices at once, with $\boldsymbol{A}$ the adjacency matrix and $\boldsymbol{X}$ the matrix of the states:

$$
\boldsymbol{X}=\tanh\big(\boldsymbol{W}^{in}\boldsymbol{U}+\hat{\boldsymbol{W}}\boldsymbol{X}\boldsymbol{A}\big)
$$

Both $\boldsymbol{W}^{in}$ and $\hat{\boldsymbol{W}}$ are **randomly initialized and never trained**.

Note the equation is **implicit**: $\boldsymbol{X}$ appears on both sides. It is a discrete-time dynamical system over the graph, so it is solved by iterating

$$
\boldsymbol{X}_t=F(\boldsymbol{U},\boldsymbol{X}_{t-1})
$$

from $\boldsymbol{X}_0=\boldsymbol{0}$ until $\lVert \boldsymbol{X}_t-\boldsymbol{X}_{t-1}\rVert<\epsilon$, or a maximum number of iterations $\nu$ is reached. Each iteration is one message-passing step, so after $t$ steps the state of $v$ covers a context of radius $t$ - same story as in [[Deep Graph Networks]], with the weights **shared** across the iterations.

Think of it as an [[RNN]] unrolled over the structure instead of over time, with null initial states for the neighbours.

## Graph Embedding Stability

The analogue of the [[Echo State Networks|Echo State Property]]: whatever the initial condition, the encoding developed for the same input graph must be **unique**, i.e. the iteration must converge to a single fixed point.

Contractivity gives it, by the Banach contraction principle. With $k$ the **maximum degree** of the graphs in the data:

$$
\textbf{sufficient:}\quad \lVert \hat{\boldsymbol{W}}\rVert\cdot k<1
\qquad\qquad
\textbf{necessary:}\quad \rho(\hat{\boldsymbol{W}})\cdot k<1
$$

with $\rho(\cdot)$ the spectral radius. A tighter version replaces $k$ with $\max\lVert \boldsymbol{A}\rVert$ over the data set, since $\lVert \boldsymbol{A}\rVert\le k$:

$$
\big(\max_{\mathcal{G}}\lVert \boldsymbol{A}(\mathcal{G})\rVert\big)\cdot\lVert \hat{\boldsymbol{W}}\rVert<1
$$

> [!important] Where the graph enters the stability
> In a plain ESN the condition is on $\rho(\hat{\boldsymbol{W}})$ alone. Here the **topology** joins in: the more connected the graph, the more signal each node receives, so the weights have to be scaled down accordingly.

$\rho(\hat{\boldsymbol{W}})\,k$ is the knob you actually set: draw $\hat{\boldsymbol{W}}$ uniformly in $[-1,1]$ and **rescale** it to the desired value.

## Readout

The node states are pooled into a graph embedding (global pooling, e.g. a sum over the vertices) and fed to a **linear** readout:

$$
y(\mathcal{G})=\boldsymbol{W}^{out}\sum_{v\in\mathcal{V}}\boldsymbol{x}(v)
$$

trained in closed form by pseudo-inversion / [[Ridge Regression]] - exactly as in [[Randomized Neural Networks]]. A fixed random projection plus a non-linearity is sometimes inserted before the readout. For node-level tasks, skip the pooling and train the readout on the node states directly.

## Deep GESN

Stack $L$ reservoir layers. Layer $i$ uses the embedding produced by layer $i-1$ as its input:

$$
\boldsymbol{u}^{(1)}(v)=\boldsymbol{l}(v),\qquad \boldsymbol{u}^{(i)}(v)=\boldsymbol{x}^{(i-1)}(v)\ \ (i>1)
$$

$$
\boldsymbol{X}^{(i)}=\tanh\big(\boldsymbol{W}^{in(i)}\boldsymbol{U}^{(i)}+\hat{\boldsymbol{W}}^{(i)}\boldsymbol{X}^{(i)}\boldsymbol{A}\big)
$$

each layer iterated to its own fixed point, each with its own stability constraint $\rho(\hat{\boldsymbol{W}}^{(i)})\,k<1$.

So a higher-order embedding is built on top of the previous one, and depth is again functional to the growth of the context. The point being made here: the advantage of layering is an **architectural effect**, since nothing in the encoder is trained.

## Hyperparameters

The reservoir is not trained, but it is **designed** - and these are the knobs to select, see [[Hyperparameter Guidelines]]:

| | |
|---|---|
| $\rho(\hat{\boldsymbol{W}})\,k$ | the stability/contractivity knob, the most important one |
| $\omega$ | input scaling of $\boldsymbol{W}^{in}$ |
| $H$ | number of reservoir units (large, sparsely connected) |
| $L$ | number of layers |
| $\epsilon,\nu$ | convergence tolerance and iteration cap |
| $\lambda$ | regularization of the readout |

**Sparse designs** (e.g. a block-diagonal reservoir) cut the cost further without losing accuracy on node classification.

## Why bother with an untrained encoder

- **Efficiency.** No backpropagation through the message-passing layers, and the cost of the encoding is the same at training and at test time. It scales to very large graphs, which is one of the open issues in [[Deep Graph Networks]].
- **A clean experiment.** It **decouples the inductive bias of message passing from the problems of end-to-end training**, so what you observe is the effect of the architecture, not of the optimizer.
- **Direct control of the dynamics.** Because $\lVert \hat{\boldsymbol{W}}\rVert$ is set by hand and not learned, you can deliberately break the contractive regime: imposing $\lVert \hat{\boldsymbol{W}}\rVert>1$ keeps the sensitivity to distant inputs from vanishing exponentially (the over-squashing bound in [[Deep Graph Networks]]), which is what makes GESN hold up on **heterophilic** node classification and resist **over-smoothing** as the number of message-passing steps grows.

The cost is the usual one of the randomized approach: the encoding is **not adapted** to the task - see the debate in [[Randomized Neural Networks]].

## Dynamic graphs: DynGESN

Extension to **discrete-time dynamic graphs**, where the edges (or the node values) change over time - infection spreading, dissemination on social networks, signals on nodes.

Now the neighbourhood is the one at time $t$, $\mathcal{N}_t(v)$, and the state recurs over **time** instead of iterating to a fixed point:

$$
\boldsymbol{x}^{(\ell)}_t(v)=\gamma_\ell \tanh\Big(\boldsymbol{W}^{in(\ell)}\boldsymbol{x}^{(\ell-1)}_t(v)+\sum_{v'\in\mathcal{N}_t(v)}\hat{\boldsymbol{W}}^{(\ell)}\boldsymbol{x}^{(\ell)}_{t-1}(v')\Big)+(1-\gamma_\ell)\,\boldsymbol{x}^{(\ell)}_{t-1}(v)
$$

- $\gamma_\ell\in(0,1]$ is a **leakage** constant: $\gamma=1$ gives the plain update, smaller values make the state integrate more slowly, so the layer reacts on a longer time scale (the same trick used for multiple time scales in [[Echo State Networks|deep ESN]]).
- It is a **spatio-temporal** graph convolution: one message-passing step per time step, so the context grows in space and in time together.

**Stability.** The contraction coefficient at time $t$ is

$$
C_t=(1-\gamma_\ell)+\gamma_\ell\,\lVert \hat{\boldsymbol{W}}^{(\ell)}\rVert\,\lVert \boldsymbol{A}_t\rVert
$$

and the sufficient condition for the echo state property involves the **geometric mean** of $\lVert \boldsymbol{A}_t\rVert$ over the graph series, instead of a single maximum degree. Since that mean is $\le k$, this is a **weaker** (so less restrictive) condition than the static one above.

**Embedding.** Sum pooling per layer, concatenated across the layers:

$$
\boldsymbol{X}^{\mathcal{G}}=\Big[\sum_{v}\boldsymbol{x}^{(1)}_t(v)\ \dots\ \sum_{v}\boldsymbol{x}^{(L)}_t(v)\Big],
\qquad
y=\boldsymbol{W}^{out}\boldsymbol{X}^{\mathcal{G}}+\boldsymbol{b}
$$

again with a linear trained readout. Compared with fully-trained temporal graph networks it reaches comparable effectiveness at a small fraction of the training cost.

## Related

[[Deep Graph Networks]], [[Echo State Networks]], [[Randomized Neural Networks]], [[Structured Domain Learning]], [[RNN]], [[Ridge Regression]], [[Cover's Theorem]].
