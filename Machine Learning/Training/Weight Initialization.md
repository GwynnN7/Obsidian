---
tags: [machine-learning, training, hyperparameter]
---

# Weight Initialization

Initialize the weights with **random values near zero**.

## What to avoid

- **All zero** $\to$ symmetric hidden layers, no symmetry breaking, the units never differentiate.
- **All equal** $\to$ same problem.
- **High values** $\to$ saturation of the activations ($f'\approx0$, tiny deltas, training crawls). See [[Activation Functions]].

## Practical ranges

- E.g. $[-0.7,+0.7]$ for standardized data.
- **Fan-in aware:** range $\times\,2/\text{fan-in}$, where fan-in is the number of inputs to a (hidden) unit.
  - Not if the fan-in is too large.
  - Not for output units, otherwise the deltas start close to zero.
- Other heuristics: orthogonal matrices, ...

## The three named schemes

They all answer the same question: **choose the variance of the initial weights so that the variance of the activations (and of the gradients) stays roughly constant from layer to layer.** Too small and the signal dies going forward while the gradient vanishes going backward; too large and it explodes or saturates.

| Scheme | Target variance | Uniform range | Made for |
|---|---|---|---|
| **LeCun** | $\operatorname{Var}(w)=\dfrac{1}{\text{fan-in}}$ | $\pm\sqrt{3/\text{fan-in}}$ | sigmoid/tanh in their linear regime; SELU |
| **Glorot / Xavier** | $\operatorname{Var}(w)=\dfrac{2}{\text{fan-in}+\text{fan-out}}$ | $\pm\sqrt{6/(\text{fan-in}+\text{fan-out})}$ | **tanh**, logistic - symmetric activations |
| **He / Kaiming** | $\operatorname{Var}(w)=\dfrac{2}{\text{fan-in}}$ | $\pm\sqrt{6/\text{fan-in}}$ | **ReLU** and variants |

- Each has a gaussian form ($\mathcal{N}(0,\operatorname{Var})$) and a uniform form - the ranges above give exactly that variance.
- **Glorot** (Glorot & Bengio, AISTATS 2010) balances the forward and the backward pass, hence the average of fan-in and fan-out. Very popular among practitioners.
- **He** doubles the variance because ReLU zeroes about half of the activations, halving the variance at every layer - see [[Activation Functions]].
- **LeCun** is the fan-in-only version - the one the classic $\times\,2/\text{fan-in}$ heuristic above is approximating.
- **Biases** go to $0$ in all three: there is no symmetry to break on a bias.

Rule of thumb: **tanh $\to$ Glorot, ReLU $\to$ He.** The deeper the network, the more this matters - see [[Deep Learning]].

> [!note]
> A **very small** NN (very few units) has a much stronger dependence on the initialization: a high initial range can make convergence slow, so use small random ranges or fan-in.

Since a NN starts near-linear with small weights and grows in capacity as the weights grow, the initialization is also the starting point of the variable-size $H$ story - see [[Neural Networks]] and [[Early Stopping]].

Related: [[Multiple Minima]], [[Hyperparameter Guidelines]].
