---
tags: [machine-learning, models, neural-networks]
---

# Echo State Networks

Reservoir Computing (RC): the [[Randomized Neural Networks|random weights]] idea applied to [[RNN]]. Also called ESN / liquid state machine / fractal machine.

The question: *can we use the recurrent state machine to encode sequences, and then only learn the output mapping?* Yes - even with randomly connected recurrent units whose weights are never trained.

## Architecture

$$
\text{input } l(t) \xrightarrow{\ W^{in}\ } \underbrace{\text{Reservoir}}_{\hat W,\ \text{untrained}} \xrightarrow{\ W^{out}\ } y(t)
$$

- **Reservoir:** a *large* set of **sparsely connected** recurrent units, untrained after random initialization.
- **Readout:** a simple feed-forward layer of **linear** units, trained by efficient linear methods - e.g. ridge regression, closed-form solution. Same story as in [[Randomized Neural Networks]].

## Echo State Property

Contractivity of the state transition function, obtained by limiting the **spectral radius** $\rho$ of the reservoir weights, i.e. stability of the dynamical system.

Consequence: the recurrent state asymptotically depends (*echoes*) only on the history of the inputs, and **not** on the initial state conditions.

## The state space: architectural bias

The reservoir acts as a projection (embedding) of the input sequences into the vector of state values $x(t)$.

- It can intrinsically discriminate among different input sequences in a **suffix-based** fashion, *without* training the recurrent weights: states are close to each other proportionally to the length of their **common suffix**.
- So Markovian tasks are easy to learn - also for a fully trained [[RNN]], but ESN are far more efficient.
- This is the **architectural bias** of recurrent models, and it is present with learning too.

Related to dynamical-system studies, chaos theory and fractals.

## Deep ESN

RC + depth: you get the power of deep RNN with the efficiency of ESN. The key questions are *why stack recurrent layers* and *what is the architectural effect of layering, independent of learning*.

Results: the deep layered organization shows advantages in terms of

- occurrence of **multiple time-scales**,
- increasing **richness of the dynamics** (hierarchical).

Frequency analysis on a linear DeepESN shows diversified magnitudes of the FFT components across layers, ordered along the hierarchy: **higher layers focus on lower frequencies**, i.e. larger time scales.

## Where it is used

Efficiency-critical settings: big data, embedded/edge devices, and anywhere the cost of BPTT is prohibitive. Hardware implementations exist via photonic reservoir computing (speed, latency, power efficiency).

The same reservoir idea extends from sequences to **graphs** - the state recurs over the neighbours instead of over time, and the stability condition picks up the degree of the graph $\to$ [[Graph Echo State Networks]], within the general framework of [[Deep Graph Networks]].

## Related

[[RNN]], [[Randomized Neural Networks]], [[Graph Echo State Networks]], [[Deep Learning]], [[Ridge Regression]], [[Structured Domain Learning]].
