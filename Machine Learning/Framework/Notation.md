---
tags: [machine-learning, reference]
---

# Notation

Consistent notation across all these notes. Local variations exist and are stated where they apply.

This note fixes the **symbols**; what the operations on them actually do is in [[Vector and Matrix Algebra]].

> [!important] Bold means vector or matrix
> Vectors and matrices are written **bold** ($\boldsymbol{x}$, $\boldsymbol{w}$, $\boldsymbol{X}$); scalars are plain ($x_i$, $w_j$, $\eta$). The same letter can be either, so the weight matters: $\boldsymbol{x}_p$ is the $p$-th pattern, $x_{p,i}$ is one number inside it.

## Data

- $\boldsymbol{X}$: data matrix, $l$ rows (patterns) $\times$ $n$ columns (features/variables/attributes/components). $p=1..l$, $i=1..n$.
- $\boldsymbol{x}$: a row of the table - example, pattern, instance, sample, input **vector**.
- $x_i$ or $x_j$ (**scalar**): component $i$ or $j$ of a pattern, with $p$ omitted.
- $\boldsymbol{x}_p$: the $p$-th pattern (vector).
- $x_{p,i}$, also $(\boldsymbol{x}_p)_i$ (**scalar**): component $i$ of pattern $p$.
- Target: typically $y_p$, or $d_p$ or $t_p$ - scalar, or $\boldsymbol{d}$ for multi-output.

- $\boldsymbol{T}$: a **tensor** - an array of numbers on a regular grid, indexed by more than two indices. A scalar is a 0-tensor, a vector a 1-tensor, a matrix a 2-tensor; a batch of RGB images is a 4-tensor.

## Functions

- $f$: the unknown target function.
- $h$: the hypothesis emitted by the learning algorithm to approximate $f$; $h_{\boldsymbol{w}}$ highlights the dependence on the parameter **vector** $\boldsymbol{w}$.
- $h(\boldsymbol{x})$ is discrete for classification, real for regression.
- $o$ (or $out$): output of a unit; $net$: weighted input to a unit. Both scalars.
- $w_{tu}$, $w_{ij}$ (**scalar**): a single connection weight.

## Loss, error and risk

- $L(h(x),d)$: **inner** loss for one pattern.
- On any finite set $D_r$ of $r$ patterns:

  $$
  E(w)=\frac1r\sum_{p=1}^{r}L(h_w(x_p),d_p).
  $$

- $R$: risk over the whole data distribution.
- $R_{emp}(h,D_r)$: empirical risk, the surrogate of $R$ on a finite set. In the SLT context $R_{emp}=R_{emp}(h,D_{TR})$ (training error); in the validation context the set is always written explicitly.
- Objective function for training, e.g. Tikhonov:

  $$
  \operatorname{Loss}(w)=\text{data error (MSE)} + \text{penalty}.
  $$

> [!important]
> Always say **which** set an empirical error is measured on - $D_{TR}$, $D_{VL}$ or $D_{TS}$. The set decides whether that number is for training, selection or assessment.

## Model-specific conventions

- **Linear models / LTU:** inputs $x$, output $h(x)$, targets $y$ (regression) or $y$/$d$ (classification).
- **Neural Networks:** if the pattern is loaded in the input layer, we use $o$ for both inputs and hidden outputs. The input to unit $t$ coming from any source $u$ through $w_{tu}$ is $o_u$ - this is the style of the [[Backpropagation]] derivation. Targets $d$ (scalar) or $\boldsymbol{d}$ (vector, multi-output).
- **SVM:** inputs $x$, output $h(x)$, targets $d$; the bias $w_0$ is denoted $b$; $l\to N$, $n\to m$, indices $i,j$ for patterns.
- **Concept learning:** $d: X\to\{0,1\}$, $h$ built with literals of propositional logic. See [[Concept Learning]].
- **RNN:** domain becomes sequences - inputs $l(t)$, outputs $o(t)$ or $y(t)$, targets $d(t)$, and $x(t)$ denotes the **state** values.

## Weight indices

$w_{ij}$ is the weight **of unit $i$**, from unit/input $j$ to unit $i$ - not the other way round. Some libraries and Wikipedia use the opposite.

Related: [[ML Framework]], [[Loss Functions]], [[Vector and Matrix Algebra]].
