---
tags: [machine-learning, models, neural-networks]
---

# Randomized Neural Networks

A **randomized approach** employs a degree of randomness as part of its constructive strategy. Randomness is not an enemy - it is a friend, if you control it.

## Randomness at different levels of ML

Most of it we have already met, scattered around:

| Level | Where |
|---|---|
| Data splitting | hold-out, k-fold, LOO - [[Cross Validation]] |
| Data generation | [[Bootstrap]], [[Bagging]], [[Boosting]], balancing |
| Learning | observation order (shuffling for [[Batch Online and Mini-Batch\|mini-batch]]), [[Weight Initialization]] |
| Hyperparameter selection | [[Random Search]] |
| Model differentiation | different inits in NN ([[Multiple Minima]]), sampling different TR sets in bagging/boosting |
| Core of the model | random neurons of Restricted Boltzmann Machines; **Random Forests** / Extremely Randomized Trees; dropout |
| Architecture | **random weights NN** - this note |

**Random Forests** in one line: a pool/ensemble of randomized decision trees, each built from samples randomly drawn from the training set, with a randomized selection of the input variables used to split the nodes $\to$ ensembling + randomness advantages. See [[Ensemble Learning]].

## Random weights NN: the basic idea

Can we exploit the MLP (or [[RNN]]) architecture **without long training cycles in the hidden layers**?

- A network with a randomly connected hidden layer, i.e. weights **fixed after random initialization**;
- train **only the output weights**, which is a linear model $\to$ pseudoinverse / ridge regression.

This sidesteps the complex and computationally demanding training algorithms normally used for NN.

Long tradition (Random Vector Functional-Link, random projection architectures), recently repopularized under names like **ELM** (Extreme Learning Machine). Roots go back to the pioneering work on the [[Perceptron]], where the projection area was fixed and only the association area was adapted; other early works had already pointed out the relevance of learning in the **output** layer w.r.t. the hidden one.

## General architecture

$$
\text{input} \longrightarrow \underbrace{\text{untrained hidden layer}}_{\text{feature representation }\varphi} \longrightarrow \underbrace{\text{readout layer}}_{\text{trained}} \longrightarrow \text{output}
$$

- The **hidden layer** non-linearly embeds the input into a high-dimensional feature space where the problem is more likely to be solvable linearly. This is a randomized (large) [[Linear Basis Expansion]], theoretically grounded on [[Cover's Theorem]].
- The **readout** combines the features for the output computation, typically with a linear model.

### To build it

1. Fill $W$ (hidden layer weight matrix) with random values, e.g. gaussian noise, using **many** units.
2. Train the readout.

### To use it

$$
o(x)=W^{out}f(Wx)
$$

Compute the hidden activations as usual for neural units, compute the outputs, then apply a threshold (or another non-linearity) if you need classification.

### To learn it

Given $l$ training examples $(x_p,d_p)$, exactly as for [[Linear Models]]: $\arg\min_{W^{out}}\text{Error}(W^{out})$ in $L_2$, by the direct method.

$$
W^{out}=H^{+}d, \qquad H=f(WX)
$$

with $H^{+}$ the pseudoinverse ([[Normal Equation]]). **Often** with L2 regularization, useful given the high number of units:

$$
W^{out}=(H^TH+\lambda I)^{-1}H^Td
$$

i.e. [[Ridge Regression]] on the hidden representation. L1 is also possible.

Reference models: **RVFL**, **ELM**, RBF networks with random centres.

## Why it works, and the debate

- A large set of hidden units non-linearly expands the dimension of the input vector, making the data more separable, and provides a sufficient basis expansion to solve the task - with a theoretical ground, including universal approximation results.
- **Extremely efficient**, since training is restricted to the last layer.
- Many variants: incremental learning adding hidden neurons (compare [[Cascade Correlation]]), pruning, ... often linked to older research under different names.
- **But:** focused models with a smaller number of *adaptive* basis functions, trained on the specific task, are still arguably "more elegant". There is a real debate here - it is worth knowing the **"random features" vs "learned features"** distinction.

## Where it is used

- **Big data**, thanks to the extreme efficiency.
- **Tiny implementations** for resource-limited devices: WSNs, IoT, edge computing.
- **Deep learning**, where training is particularly demanding: deep feed-forward randomized NN are stacks of RVFL/ELM modules; randomized CNN approaches exist too.
- **Recurrent** randomized NN, where efficiency of training is even more demanding than for feed-forward MLP, and where the bias introduced has a solid ground in dynamical-system analysis (stability, learnability) $\to$ [[Echo State Networks]].

## Related

[[Neural Networks]], [[Cover's Theorem]], [[Linear Basis Expansion]], [[Echo State Networks]], [[Deep Learning]].
