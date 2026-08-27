---
tags: [machine-learning, moc, essentials]
---

# ML Essentials

The whole subject on one page, in the order that makes it make sense. Every stage is a question, a couple of notes, and the one thing to keep.

The visual version of the same path is [[ML Map]]. The folder-by-folder view is [[Machine Learning]].

## The five lines

1. We **approximate an unknown function** from finite, noisy data.
2. That is an ill-posed problem, so we must **assume something**: a hypothesis space plus a preference inside it.
3. The only real trade-off is **fitting capability vs complexity**.
4. None of it means anything unless the data used to **choose** and the data used to **judge** are kept apart.
5. Everything else - linear, local, kernel, neural, deep, structured - is just a different **choice of bias**.

## The path

### 1. The frame

*What are the pieces of any ML system?* Data, task, model, learning algorithm, validation.

$\to$ [[ML Framework]], [[Data and Preprocessing]], [[Notation]]

**Keep:** the model defines $H$, the algorithm searches $H$, validation says whether the search was worth anything.

### 2. Why learning can work at all

*How can finite data say anything about unseen data?* Only with assumptions.

$\to$ [[Hypothesis Space and Inductive Bias]], [[Concept Learning]], [[Inductive Bias]], [[No Free Lunch]]

**Keep:** a learner with no bias is a lookup table. This is proved, not asserted.

### 3. What we measure

*What does "good" mean?* A loss per pattern, averaged into a risk - and the risk we care about is on data we have never seen.

$\to$ [[Loss Functions]], [[Generalization and Risk]], [[Overfitting]] / [[Underfitting]]

**Keep:** $R_{emp}$ is what you can compute, $R$ is what you want. They are not the same number.

### 4. The rigid and the local

*The two simplest hypothesis spaces, and their opposite failure modes.*

$\to$ [[Linear Models]], [[Linear Basis Expansion]], [[Ridge Regression]] | [[k-NN]], [[Curse of Dimensionality]]

**Keep:** linear = high bias, low variance, decided at training. k-NN = low bias, high variance, decided at prediction, and it dies in high dimension.

### 5. Making the basis functions adaptive

*What if the features were learned instead of chosen?* That is a neural network.

$\to$ [[Neural Networks]], [[Perceptron]], [[Activation Functions]], [[Universal Approximation]]

**Keep:** the hidden layer re-represents the input so the output layer's job becomes easy. Everything after this is variations on that sentence.

### 6. Actually finding the weights

*How is the search through $H$ performed?* Gradient descent, with backprop to get the gradient.

$\to$ [[Gradient Descent]], [[Backpropagation]], [[Batch Online and Mini-Batch]], [[Optimizers]], [[Learning Rate]], [[Weight Initialization]]

**Keep:** we do **not** want the global minimizer of $R_{emp}$ - that is an overfitting solution. Optimization is not the goal.

### 7. Keeping complexity in check

*How do we stop it from fitting the noise?*

$\to$ [[Regularization]], [[Early Stopping]], [[Dropout]], [[Batch Normalization]]

**Keep:** $\lambda$ is not a training-stability knob, it is a **complexity** knob, and it is always chosen on validation data.

### 8. Why the trade-off exists

*The same fact in two languages.*

$\to$ [[Statistical Learning Theory]], [[VC Dimension]], [[Structural Risk Minimization]] | [[Bias-Variance Tradeoff]]

**Keep:** $R\le R_{emp}+\epsilon(1/l,\text{VC},1/\delta)$ and $\text{bias}^2+\text{variance}+\text{noise}$ are the same story: fit the data, but not too much.

### 9. Doing it honestly

*How do you get a number you are allowed to believe?*

$\to$ [[Model Selection and Assessment]], [[Validation Schemas]], [[Cross Validation]], [[Validation Pitfalls]], [[Hyperparameter Search]]

**Keep:** the gold rule below. Most of the runtime of a real project is here, not in training.

### 10. Maximum margin

*A different way to control complexity: not the number of parameters, the geometry.*

$\to$ [[SVM]], [[Kernel Methods]], [[SVM Regression]], [[Cover's Theorem]]

**Keep:** convex problem, sparse solution, complexity ruled by the **margin** and not by the dimension of the feature space. And the kernel separates the background knowledge from the solver.

### 11. Putting knowledge into the architecture

*Three ways past the plain MLP.*

$\to$ [[CNN]] (space), [[RNN]] (time), [[Randomized Neural Networks]] / [[Echo State Networks]] (don't train it), [[Attention and Transformers]]

**Keep:** weight sharing is regularization by architecture - fewer free parameters, same connections.

### 12. Depth, and what the layers learn

*Why stack them?* Not for size - for compactness and for the representation.

$\to$ [[Deep Learning]], [[Representation Learning]], [[Distributed Representation]]

**Keep:** depth is a bias (the target is a composition of simpler functions). When it fits, you get an exponential saving; when it does not, [[No Free Lunch]] still applies.

### 13. When the input is not a vector

$\to$ [[Structured Domain Learning]], [[Deep Graph Networks]], [[Graph Echo State Networks]]

**Keep:** move the models to the data, not the data to the models.

### 14. Without a teacher

$\to$ [[Unsupervised Learning]], [[Cluster Analysis]], [[Vector Quantization]], [[K-Means]], [[SOM]]

**Keep:** with no target, the **metric is the whole inductive bias** - and evaluating the result is largely subjective.

### 15. In practice

$\to$ [[Hyperparameter Guidelines]], [[ML Workflow]]

**Keep:** preprocessing and validation decide more projects than the choice of model does.

## The rules you don't break

> [!important] Gold rule
> $D_{TR}$ trains; $D_{VL}$ selects; $D_{TS}$ estimates the final risk. Once a set has influenced a design choice, it is no longer an unbiased test set. See [[Validation Pitfalls]].

> [!important] The trade-off
> Fitting capability *vs* model complexity. Two languages, one fact: [[Statistical Learning Theory]] and [[Bias-Variance Tradeoff]].

> [!important] Bias is necessary
> A learner with no [[Inductive Bias]] cannot generalize. Proved in [[Concept Learning]].

> [!warning] The three classic mistakes
> Scaling or selecting features **before** splitting; reporting the validation number as if it were a test number; comparing models on a single split.

## Notation cheat-sheet

- $l$ examples, $n$ input features; $\boldsymbol{x}_p$ pattern $p$, $d_p$ or $y_p$ its target.
- $h_{\boldsymbol{w}}(\boldsymbol{x})$: output of the model with parameters $\boldsymbol{w}$; $f$: the unknown target.
- $L(h(\boldsymbol{x}),d)$: inner loss; $R_{emp}$: empirical risk; $R$: true risk.
- $\operatorname{Loss}=$ data error $+$ penalty; **Error** $=$ the data term alone.
- **Bold** means vector or matrix; plain means scalar. Full reference: [[Notation]].

## If you only read ten notes

[[ML Framework]] · [[Hypothesis Space and Inductive Bias]] · [[Generalization and Risk]] · [[Linear Models]] · [[Neural Networks]] · [[Backpropagation]] · [[Regularization]] · [[Bias-Variance Tradeoff]] · [[Model Selection and Assessment]] · [[Hyperparameter Guidelines]]
