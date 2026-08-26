---
tags:
  - machine-learning
  - subject
  - moc
---

# Machine Learning

Machine Learning Hub. [[ML Framework]] is the starting point for a logic order.
## The map

| Folder | What is in it |
|---|---|
| [[ML Framework\|Framework]] | data, tasks, paradigms, loss, notation, workflow |
| [[Theory]] | inductive bias, generalization, SLT, bias-variance, ensembles |
| [[Models]] | linear, k-NN, neural networks |
| [[Training]] | gradient descent, backprop, all the training knobs |
| [[Validation]] | model selection and assessment, CV, metrics |
| [[Hyperparameter Guidelines\|Guidelines]] | the practical do / don't reference |
| [[To Do]] | PDFs not covered yet |

## The thread

1. **What is the problem** - [[ML Framework]], [[Loss Functions]], [[Generalization and Risk]].
2. **Why it works** - [[Concept Learning]], [[Hypothesis Space and Inductive Bias]], [[No Free Lunch]].
3. **Simple Models** - [[Linear Models]] (rigid), [[k-NN]] (flexible but local).
4. **Flexible + Controllable** - [[Neural Networks]], [[Perceptron]], [[Backpropagation]].
5. **Specialized architectures** - [[CNN]] (space), [[RNN]] (time), [[Randomized Neural Networks]] (skip training the hidden layer).
6. **Maximum margin** - [[SVM]], [[Kernel Methods]], [[SVM Regression]].
7. **The theory of the trade-off** - [[Statistical Learning Theory]], [[VC Dimension]], [[Structural Risk Minimization]], [[Bias-Variance Tradeoff]].
8. **How to do it right** - [[Validation]], [[Model Selection and Assessment]], [[Cross Validation]].
9. **How to do it in practice** - [[Hyperparameter Guidelines]].
10. **Still to do** - [[Deep Learning]], [[SOM]], [[Structured Domain Learning]].

## The three things to never forget

> [!important] Gold rule
> $D_{TR}$ trains; $D_{VL}$ selects; $D_{TS}$ estimates the final risk. Once a set influences a design choice, it is no longer an unbiased test set. See [[Validation Pitfalls]].

> [!important] The trade-off
> Fitting capability *vs* model complexity. Same idea in two languages: $R\le R_{emp}+\epsilon(1/l,VC,1/\delta)$ ([[Statistical Learning Theory]]) and $\text{bias}^2+\text{variance}+\text{noise}$ ([[Bias-Variance Tradeoff]]).

> [!important] Bias is necessary
> A learner with no [[Inductive Bias]] is a lookup table: it cannot generalize. Proved in [[Concept Learning]].

## Notation cheat-sheet

- $l$ examples, $n$ input features; $x_p$ pattern $p$, $d_p$ or $y_p$ its target.
- $h_w(x)$: output of the model with parameters $w$; $f$: the unknown target.
- $L(h(x),d)$: inner loss; $R_{emp}$: empirical risk; $R$: true risk.
- $\operatorname{Loss}=$ data error $+$ penalty; **Error** $=$ the data term alone.
- Full reference: [[Notation]].

Part of [[Artificial Intelligence]] - the hub holds the concepts shared with the other subjects.
