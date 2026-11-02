---
tags: [artificial-intelligence-fundamental, models]
---

# Cellular Automata

A simple mathematical model for simulating **self-replication**, and the standard example of complex behaviour emerging from local rules.

Context: the **logistic map** shows the transition from ordered behaviour to chaos inside a purely deterministic system.

## Structure

- A **discrete space** (an array of cells) and **discrete time**.
- Each cell holds one discrete variable.
- Dynamics governed by a **local update rule**.

**Applications:** circuit design, pseudorandom number generation, cryptography, image processing (segmentation of high-resolution images), image compression.

## Elementary CA (ECA)

One-dimensional CA with binary-valued cells. Update rule is a Boolean function of the cell and its two neighbours:

$$
s_j^{t+1}=f(s_{j-1}^t,\ s_j^t,\ s_{j+1}^t)
$$

- **Constraint:** an initial zero-state must map to zero (last digit of the rule = 0).
- **Symmetry:** reflection symmetry, leading to isotropy - consistent behaviour regardless of direction.
- Typically initialized from **white noise**; the whole state is characterized by a single parameter $p$.
- Often produces an ordered system with **self-organization** and self-similarity at scale (fractals).

### Wolfram classification

| Type | Evolution |
|---|---|
| **1** | to a static state |
| **2** | to a periodic structure |
| **3** | to chaotic patterns - pseudo-randomness |
| **4** | to complex patterns, at the **edge of chaos** |

## Game of Life

2D CA, two states per cell (dead/alive), eight neighbours. Rule **23/3**:

- a living cell **survives** with exactly two or three live neighbours, otherwise it dies;
- a dead cell is **born** with exactly three live neighbours.

## Lenia (continuous CA)

Local update rule defined by a **potential distribution**, a **growth distribution** and a **kernel**.

Extended features: multi-dimensional execution for more stable forms; multiple kernels, each with a relative radius and growth mapping; multiple **channels** (interacting "worlds"). Evolutionary search over them is challenging - filtered by survival, human filtering and random changes.

Observed results: self-organizing structures (**solitons**), self-replication (a soliton splits and the parts repel), coordination between channels.

## Neural Cellular Automaton (NCA)

Studies the mechanisms of **organism shape generation**: self-organization, robustness to perturbations, controlled growth and cessation.

- **Cell state:** a real-valued vector of 16 numbers. The first three channels are RGB, the fourth is alpha. Alpha $>0.1$ = living cell, $\le0.1$ = growing cell. Dead cells (no living neighbour) have their state set to 0.
- **Perception:** a fixed-kernel neighbourhood **convolution** produces a 48-number perception vector - see [[CNN]].
- **Update rule:** a small [[Neural Networks|neural network]] (~8k parameters) returning a state update. The network is **shared across all cells** (weight sharing again), and the update is applied randomly with 50% probability per cell. Iterative, typically from a single living cell.
- **Stability:** either set the target pattern as an attractor (computationally expensive), or learn from a **pool of states** updated over time with intermediate states.
- **Regeneration after trauma:** damage the pool states (zero out a random circular region) before use.
- **Self-classifying digits:** each cell predicts the class and the final prediction comes from cellular agreement; immutable input channels plus 19 trained channels, the last 10 being class probabilities (~25k parameters).

Related: [[Artificial Life]], [[CNN]], [[Neural Networks]].
