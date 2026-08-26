---
tags: [machine-learning, models, neural-networks]
---

# Neural Networks

For us: **ANN = a flexible ML tool**, i.e. a mathematical function $h(x)$ with special properties. The biological inspiration matters historically, but the computational and algorithmic properties are what count here.

Why they are interesting:

- learn from examples, universal approximators ([[Universal Approximation]]);
- deal with noise and incomplete data - performance degrades gracefully;
- handle continuous and discrete data, regression and classification;
- a **paradigm**, i.e. a wide set of models (subsymbolic approaches).

## The unit

Node / neuron / unit $t$, with inputs from an external source or from other units:

$$
net_t(x)=\sum_u w_{tu}x_u,\qquad o_t(x)=f_t(net_t(x)),
$$

with $x_0=1$ and $w_{t0}$ the bias. $f$ is the [[Activation Functions|activation function]].

> [!note] Index convention
> $w_{ij}$ is the weight **of unit $i$**, from input/unit $j$ to unit $i$. Some libraries (and Wikipedia) use the opposite - see [[Notation]].

## From McCulloch & Pitts to the MLP

McCulloch & Pitts (1943): binary units, positive (excitatory) / negative (inhibitory) weights, a unit fires when the weighted sum plus bias exceeds zero. Basically the LTU / [[Perceptron]].

- A single unit can represent **AND, OR, NOT** (hence NAND, NOR):
  - AND on 2 vars: $w_1=w_2=1$, $w_0=-1.5$.
  - OR on 2 vars: $w_1=w_2=1$, $w_0=-0.5$.
- $\Rightarrow$ every Boolean function can be represented by *some* network of perceptrons, and **two levels are enough** (more can be more efficient).
- **But a single unit cannot** (Minsky & Papert 1969): XOR is not linearly separable.

### XOR with two layers

Hidden layer: $h_1=\text{AND}(x_1,x_2)$ with $w_0=-1.5$, $h_2=\text{OR}(x_1,x_2)$ with $w_0=-0.5$; output $=\lnot h_1 \land h_2$.

By De Morgan: $\lnot(a\land b)=\lnot a\lor \lnot b$, so

$$
\lnot h_1\land h_2=(\lnot a\lor\lnot b)\land(a\lor b)=(\lnot a\land b)\lor(a\land\lnot b)=a \oplus b .
$$

The point is not the trick but the **internal representation**: $(1,0)$ becomes $(h_1,h_2)=(0,1)$, and in that new space the problem *is* linearly separable. Developing high-level hidden features is the key factor of NN, and the road to representation learning and [[Deep Learning]]. The general principle behind "expand, then separate linearly" is [[Cover's Theorem]].

## Two views of an MLP

**A. A network of units** organized in layers: the input layer only loads (copies) the pattern $x$ - input units do not compute $net$ and $f$; each hidden layer projects onto the next; the last one gives $h(x)$.

**B. A flexible function** - nested nonlinear functions:

$$
h(x)=f_k\Big(\sum_j w_{kj}\,f_j\big(\sum_i w_{ji}x_i\big)\Big).
$$

Compare with a linear model $h(x)=\sum_i w_ix_i$. Note $h$ is **non linear in the parameters** $w$ - hence a non-convex optimization problem.

Components of any NN model: **unit** (net, activation), **architecture** (number of units, topology, layers), **learning algorithm**.

## Feedforward vs recurrent

- **Feedforward:** direction input $\to$ output. Processing: load the pattern, compute layer by layer up to the output, then compute the error at the output level.
- **Recurrent:** feedback loops in the topology $\to$ dynamical properties, a memory of past computations, able to process sequences and structured data. See [[RNN]].

## Beyond the plain MLP

The MLP is fully connected and fully trained. Three ways to change that:

- **[[CNN]]** - restrict the architecture (local receptive fields) and share weights across space, to put prior knowledge about images into the network.
- **[[RNN]]** - add feedback loops and share weights across time, to get a memory of the past.
- **[[Randomized Neural Networks]]** - don't train the hidden layer at all; keep it random and train only a linear readout.

## NN as adaptive [[Linear Basis Expansion]]

LBE uses **fixed** $\phi_k$ chosen before seeing the data. A NN uses

$$
\phi_j(x,w)=f_j\Big(\sum_i w_{ji}x_i\Big),
$$

i.e. the basis functions themselves are **adapted to the data** by fitting the $w$ inside $\phi$. Same type of basis function for all the terms (given by the activation function).

This is the important connection: $h(x)$ is a nonlinear function of weighted sums of nonlinearly transformed linear models, **plus adaptivity**.

## Expressive power and VC-dim

- Depends on the number of units **and** on the configuration (architecture).
- Also on the **size** of the weights:
  - weights $=0$ $\to$ minimal VC-dim,
  - small weights $\to$ units work in the linear part of the activation $\to$ small VC-dim,
  - larger weights $\to$ more complex model.
- Since we start with weights near zero and they grow during training, a NN is a **variable-size hypothesis space**: the VC-dim increases during training. This is why stopping matters - see [[Early Stopping]] and [[Structural Risk Minimization]].

## Inductive bias

Smoothness: small input variations $\to$ small output variations (e.g. a locally limited first derivative). Sensible, because a non-smooth target (a random number generator) cannot be generalized at all.

## Learning: the credit assignment problem

For a single [[Perceptron]] the error signal is directly measurable. In an MLP we do **not** know the desired response of a hidden unit - which credit goes to it? Plus the model is non-linear in $w$.

- Judged too difficult by Minsky-Papert (1969); solved by [[Backpropagation]], popularized by Rumelhart, Hinton, Williams (PDP book, 1986) - the renaissance of NN.
- The **loading problem** ("is there a set of weights consistent with the examples?") is NP-complete (Judd, 1990). In practice networks train in reasonable time, but an optimal solution is not guaranteed.
- Requirements: differentiable loss, differentiable activation functions, and a network to follow the information flow.

## Perceptron vs LMS/Adaline

Two historical ways of training a single unit:

| | Perceptron | LMS / Adaline |
|---|---|---|
| Output during training | $\operatorname{sign}(w^Tx)$ | $w^Tx$ (linear) |
| Objective | minimize misclassifications | minimize squared error |
| Update term $\delta$ | $d-\operatorname{sign}(w^Tx)$ | $d-w^Tx$ |
| Convergence | finite steps if linearly separable, to a perfect classifier; otherwise no convergence (cycles) | asymptotic, also for non-separable problems |
| Classification errors | zero if separable | not necessarily zero, even if separable |
| Extension to networks | difficult | easy, gradient based $\to$ MLP |

- Perceptron update with bipolar labels on a misclassified pattern: $w_{new}=w+\eta\,d\,x$ (Hebbian form).
- [[Delta Rule]] (error correction form): $w_{new}=w+\eta(d-out)x$.
- Using a sigmoid $f_\sigma$ bridges the two: $\delta_p=(d_p-o(x_p))f'_\sigma$ goes to zero for correct, saturated outputs, so LMS starts behaving like the perceptron rule too.

## Related

- [[Perceptron]], [[Activation Functions]], [[Universal Approximation]]
- [[Backpropagation]], [[Training]]
- [[Regularization]], [[Hyperparameter Guidelines]]
- [[Cascade Correlation]], [[Multiclass Classification]]
- [[CNN]], [[RNN]], [[Randomized Neural Networks]] - the specialized architectures
- [[SVM]] - fixed (kernel) basis functions instead of adaptive ones
