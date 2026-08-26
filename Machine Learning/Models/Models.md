---
tags: [machine-learning, moc, models]
---

# Models

From rigid to flexible, each step adding a piece: linear $\to$ basis expansion $\to$ k-NN (flexible but local) $\to$ neural networks (flexible **and** compact, with complexity control) $\to$ specialized architectures.

## Continuous $H$, parametric, eager

- [[Linear Models]] - LMS, [[Normal Equation]], gradient descent, LTU classifier.
- [[Linear Basis Expansion]] - fixed nonlinear features, still linear in $w$.
- [[Ridge Regression]] - the regularized linear model.
- [[Multiclass Classification]] - 1-of-K, OVA, AVA, softmax.

## Instance-based, lazy, local

- [[k-NN]] - store the data, decide at prediction time.
- [[Curse of Dimensionality]] - why locality breaks in high $n$.
- [[Bayes Optimal Classifier]] - the theoretical best, which k-NN approximates.

## Neural Networks: the core

- [[Neural Networks]] - units, MLP, the two views (network / nested function).
- [[Perceptron]] - the single unit and its convergence theorem.
- [[Activation Functions]] - linear, threshold, logistic, tanh, softmax, ReLU, RBF.
- [[Universal Approximation]] - what one hidden layer can do.
- [[Cascade Correlation]] - let the algorithm decide the number of units.

## Maximum margin

- [[SVM]] - margin, support vectors, primal/dual, soft margin.
- [[Kernel Methods]] - the kernel trick, valid kernels, and life beyond SVM.
- [[SVM Regression]] - the $\epsilon$-insensitive tube.

## Neural Networks: architectures

Three ways of going beyond the plain MLP:

| | Idea | Note |
|---|---|---|
| **Prior knowledge in the architecture** | local connections + weight sharing over space | [[CNN]] |
| **Memory / state** | feedback loops, weight sharing over time | [[RNN]] |
| **Don't train the hidden layer** | random projection + trained linear readout | [[Randomized Neural Networks]] |

And their intersections:

- [[Echo State Networks]] - random weights **and** recurrent.
- [[Graph Echo State Networks]] - random weights **and** recurrent **and** on graphs.
- [[Attention and Transformers]] - keep the past instead of compressing it.

All three lean on the same principle: [[Cover's Theorem]] for the random/expansion side, and weight sharing as **architectural regularization** for the CNN/RNN side.

## Going deep

Stack the layers and the story becomes about the **representation** rather than the units:

- [[Deep Learning]] - why depth helps (no-flattening, compositionality), and the techniques that make it trainable.
- [[Representation Learning]] - autoencoders, pre-training, transfer learning.
- [[Distributed Representation]] - one-hot vs dense codes, word embedding, non-local generalization.

## Beyond vectors

- [[Structured Domain Learning]] - sequences, trees and graphs: move the models to the data.
- [[Deep Graph Networks]] - message passing, and why it has to be deep.
- [[Graph Echo State Networks]] - the same, with the message passing left untrained; and DynGESN for graphs that change over time.

## Without a teacher

No target, only the metric:

- [[Vector Quantization]] - the framework: codebook, Voronoi cells, quantization error.
- [[K-Means]] - the plain algorithm, on-line and batch.
- [[SOM]] - the same, plus a topology on the prototypes, so you can look at the result.

## Comparison table

| | Linear | k-NN | NN (MLP) | Randomized NN | SVM |
|---|---|---|---|---|---|
| Timing | eager | lazy | eager | eager | eager (but keeps the SVs) |
| $H$ | rigid, linear boundaries | very flexible, local | flexible, global | flexible, random features | linear in a feature space |
| Basis functions | none | none | **adaptive** (learned) | **random**, fixed | **fixed**, chosen by the kernel |
| Parameters | $n+1$ | $\approx l/k$ effective | many weights, variable-size $H$ | only the readout is trained | sparse $\alpha$: only the SVs |
| Optimization | convex | none | **non-convex** | convex (linear readout) | **convex** QP |
| Complexity ruled by | $n$, $\lambda$ | $k$ | #units, $\lambda$, epochs | #units, $\lambda$ | the **margin**, $C$, kernel params |
| Cost | at training | at prediction | at training (high) | at training (very low) | scales with $N$ (dual) |

Related: [[Hypothesis Space and Inductive Bias]], [[No Free Lunch]].
