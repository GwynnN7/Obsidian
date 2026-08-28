---
tags: [machine-learning, models]
---

# RNN

Recurrent Neural Networks. A different **category of architecture**: feedback loops are added to the topology.

- Feedforward: direction input $\to$ output.
- Recurrent: self-loop connections give the network **dynamical properties**, keeping a memory (**states**) of the past computations.
- This extends the representation capability to **sequences** (and structured data).
- Neurobiological plausibility: the nervous system *is* a recurrent NN.

## The result

An RNN is a **state transition system** whose transition function is a neural network:

$$
x(t)=\tau\big(x(t-1),\,l(t)\big),
\qquad
y(t)=g\big(x(t),\,l(t)\big),
\qquad
x(0)=0
$$

For a single recurrent unit that is

$$
x(t)=f\big(w^T l(t)+\hat w\,x(t-1)+\theta\big)
$$

| symbol | what it is |
|---|---|
| $l(t)$ | input at step $t$ |
| $x(t)$ | **state** - a summary of the past, continuous valued |
| $\tau$ | state transition (next-state) function, realized by a NN |
| $g$ | output function / readout |
| $\hat w$ | the **recurrent** weight; $w$ the input weights, $\theta$ the bias |
| $f$ | sigmoidal activation |

Three assumptions make it work, and each one earns something:

| assumption | what it buys |
|---|---|
| **causality** - the output at $t_0$ depends only on inputs at $t<t_0$ | an internal state is possible at all |
| **stationarity** - $\tau$ is the same at every $t$ | one **fixed-size** model handles sequences of any length |
| **adaptivity** - $\tau$ is a NN with free weights | the encoding of the past is *learnt*, not designed |

Trained by **unfolding through time** into a feedforward network with shared weights and running [[Backpropagation]] on it (BPTT, or RTRL for the same gradient computed on-line).

## When you need it

Whenever the output depends on the **history** of the inputs (dynamical models, e.g. time), or the domain is made of **varying-size sequences**:

- dynamical processes, signal processing (filters, control), robotics;
- language: speech recognition, NLP, formal languages, information retrieval;
- vision, reasoning about temporal events;
- temporal series: financial forecasting;
- genomics/proteomics (structure prediction from the sequence of amino acids);
- activity recognition from sensor data.

Also, for fun: music composition, text/speech generation.

## The sequence domain

Data: a discrete sequence of vectors in serial order, e.g. $l_t=[1,0,1,0.7]$, with **length that can vary**. See [[Data and Preprocessing]] for the general structured-data picture.

### Transduction tasks

A transduction is a mapping *input sequence $\to$ output value or sequence*:

| Task | Output |
|---|---|
| **Sequence classification / recognition** | one output at the end |
| **IO-transduction** (input-output isomorphic) | one output per input step |
| **Next-step prediction** | the next element; with a single sequence this is **autoregressive** - the basis of modern generative models |
| **Sequence generation** | a whole output sequence |

## Two ways of getting memory

### 1. Finite memory / spatial approach (TDNN)

Input-delay neural networks: a finite-size **shift register**, i.e. a sliding window over the sequence, feeding an MLP. If the buffer contains enough information, plain MLP training solves it.

- **But:** you must know the window size in advance, and the number of weights depends on the buffer size.
- This is the same idea [[CNN]] extends to 2D images.

### 2. RNN

A recurrent unit with feedback in the architecture, using the current input **plus** state information:

$$
x(0)=0,\qquad
x(t)=\tau(l(t),x(t-1))=f\big(w^T l(t)+\hat w\,x(t-1)+\theta\big)
$$

with $f$ a sigmoidal activation, $\hat w$ the **recurrent weight**, $\theta$ the bias, and $q^{-1}$ the unit-time delay.

- The **state summarizes the past** (the context) and changes with the stream of inputs.
- The encoding of the past is **adaptive**, because those weights are free parameters of the model.

## As a state transition system

Reading the two equations at the top:

- $\tau$ is the **state transition (next-state) function**, realized by a NN.
- The state has **continuous** values, and $x(t)$ can be a set of states (a network of recurrent units).
- Same graphical model family as HMM, but with continuous adaptive states.

### The three assumptions

1. **Causality** - the output at time $t_0$ depends only on inputs at $t<t_0$. Necessary and sufficient for an internal state.
2. **Stationarity** - time invariance after training: $\tau$ is the same at every $t$, regardless of the length of the sequence. This is what lets a **fixed-size model** process data of variable length.
3. **Adaptivity** - the transition functions are realized by NN with free weights, so they are learnt from data.

Even a simple RNN is already powerful: **universal approximators of non-linear dynamical systems**, and **Turing equivalent** (any automaton can be simulated). Seen as a non-autonomous dynamical non-linear system, it connects to chaos and fractal theory.

## Learning: unfolding through time

Unroll the same model through the input sequence to build a feedforward MLP - the **encoding network** - equivalent to the RNN on those $k$ steps:

- one model-replica per input step;
- **stationarity $\to$ weight sharing** across the unfolded layers (the same trick as [[CNN]], along time instead of space);
- a *different* encoding network for each input sequence;
- and now we can apply [[Backpropagation]] to it.

Two algorithms compute the same gradient of the output errors over the unfolded network, in different styles:

- **BPTT** - back-propagation through time;
- **RTRL** - real time recurrent learning.

> [!warning] Vanishing gradient
> The unfolded network is **deep**, so the gradient can vanish across it, which makes **long-term dependencies** hard to learn.

## Variants

- **LSTM** (long short-term memory): uses **gate units** to select what flows through (past and gradient).
  > [!note]
  > Born to address the need for *long* memory - not suitable for short series. Despite their popularity, do not use them acritically.
- **GRU** (gated recurrent units): a simplified LSTM.
- **BRNN** (bi-directional): considers both the left and the right context.
- Often fused with deep NN.
- Other ways to attack the vanishing gradient: Hessian-free optimizers, pre-training techniques.
- [[Echo State Networks]] - don't train the recurrent part at all.
- [[Attention and Transformers]] - drop recurrence entirely.

## Other models for sequences

SOM/unsupervised NN for sequential domains ([[SOM]]), [[Hidden Markov Models]] (the probabilistic cousin: a discrete hidden state instead of a learned continuous one), [[Attention and Transformers]], [[Echo State Networks]], distance-based models (string matching), kernels for strings, grammatical inference, ILP.

## Towards structured data

Can the recurrent approach be extended to rooted **trees**? Yes - **Recursive Neural Networks**:

- bottom-up encoding of input trees,
- extending the state to the children of each vertex.

Unfolding now happens **through the structure** instead of through time, and if the tree changes, the encoding changes accordingly. Weight sharing again: the units are the same for all the vertices of a tree and for all the trees in the data set.

$\to$ [[Structured Domain Learning]]

## Notation

The domain moves from vectors to sequences: inputs $l(t)$, outputs $o(t)$ or $y(t)$, targets $d(t)$, and **$x(t)$ denotes the state values** (not the input). See [[Notation]].

## Related

[[Neural Networks]], [[CNN]], [[Backpropagation]], [[Echo State Networks]], [[Attention and Transformers]], [[Deep Learning]], [[Structured Domain Learning]].
