---
tags: [machine-learning, theory]
---

# Universal Approximation

- A **single hidden-layer** network with logistic activations can approximate arbitrarily well every continuous function on hypercubes, provided enough hidden units.
- An MLP can approximate arbitrarily well every input-output mapping, provided enough units in the hidden layers.

Formally: given $\epsilon$, there exists $h(x)$ such that $|f(x)-h(x)|<\epsilon$ for all $x$ in the hypercube, with

$$
h(x)=f_k\Big(\sum_j w_{kj}f_j\big(\sum_i w_{ji}x_i\big)\Big),\quad f_k \text{ linear}.
$$

(Image it as generalizing the approximation by a finite Fourier series.)

> [!warning] It is an existence theorem
> It gives neither the learning algorithm nor the number of units.

## Two issues left open

1. **How to learn** $\to$ [[Backpropagation]].
2. **How to decide the architecture** $\to$ [[Model Selection and Assessment]], [[Cascade Correlation]].

## How many layers?

- One hidden layer is sufficient *in general*, but nothing guarantees a **small** number of units.
- Bounds on that number exist for many function families.
- There are also "**no flattening**" results (about efficiency, not expressivity): for some functions a single hidden layer would need an exponential number of units w.r.t. the input dimension, while more layers help - for the number of units/weights and/or for learning the approximation. The clearest example is the parity function, in [[Deep Learning]].
- Open question: is a deep MLP easy to optimize? $\to$ [[Deep Learning]].

Related: [[Neural Networks]], [[Structural Risk Minimization]].
