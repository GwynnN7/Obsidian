---
tags: [machine-learning, framework]
---

# Loss Functions

The **inner loss** $L(h(x),d)$ measures the distance between the model output and the target for **one** pattern. The error/risk is the expected value of $L$ over a set of data:

$$
E(w)=\frac1r\sum_{p=1}^{r}L(h_w(x_p),d_p).
$$

$L$ changes with the task.

## Regression

Squared error:

$$
L(h_w(x_p),d_p)=(d_p-h_w(x_p))^2 .
$$

Averaged over the data set this is the [[Mean Squared Error]].

## Classification

0/1 loss:

$$
L(h_w(x_p),d_p)=\begin{cases}0 & \text{if } h_w(x_p)=d_p\\ 1 & \text{otherwise}\end{cases}
$$

The mean is the fraction of misclassified patterns (20 out of 100 $\to$ 20% error, 80% accuracy).

- Piecewise-constant losses are hard to optimize, so for training we usually replace them with a smooth surrogate (MSE, or cross-entropy). See [[Linear Models]].

## Clustering / vector quantization

$$
L(h(x_p))=(x_p-h(x_p))\cdot(x_p-h(x_p)),
$$

i.e. the squared distance of the pattern from the centroid of its cluster. See [[Cluster Analysis]].

## Density estimation

$$
L(h(x_p))=-\ln(h(x_p)),
$$

related to maximizing the (log) likelihood.

## Loss vs Error

> [!important]
> Once a penalty term enters the picture, keep the two names apart:
> - **Loss**: the objective actually minimized during training, $\text{Loss}(w)=\text{data error}+\text{penalty}$.
> - **Error / Risk**: the data term alone, the number you report in tables and plots because it is what the user cares about.
>
> See [[Regularization]].
