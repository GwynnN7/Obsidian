---
tags: [machine-learning, guidelines, practical, reference]
---

# Hyperparameter Guidelines

Reference page for setting up backprop training, plus the validation rules that go with it. **(!)** marks what is basically mandatory.

## Philosophy

- Two general problems with NN: the model is often **over-parametrized**, and the optimization problem is **not convex and potentially unstable**.
- Most of these are heuristics, but some have a theoretical ground and most can be framed in a more general framework (regularization).
- The optimization technique only matters when there are actual obstacles to training. **Validation matters more than the optimization on the training set** - optimizing on training data is not the purpose.

---

## 1. Weight initialization

**DO**

- **(!)** random values **near zero**, e.g. $[-0.7,+0.7]$ for standardized data;
- consider the **fan-in**: range $\times\,2/\text{fan-in}$;
- or Glorot-Bengio: 0 for the biases, uniform in $[-1/\sqrt{\text{fan-in}},+1/\sqrt{\text{fan-in}}]$.

**DON'T**

- all zero (symmetry, hidden units never differentiate);
- all equal (same problem);
- high values (**saturation** $\to$ $f'\approx0$ $\to$ training crawls);
- fan-in scaling if the fan-in is too large, or on the **output units** (deltas would start close to zero).

**Watch out:** a very small NN (very few units) depends much more strongly on the initialization.

$\to$ [[Weight Initialization]]

---

## 2. Batch / on-line / mini-batch

|              | Batch                       | On-line (SGD)                                                | Mini-batch               |
| ------------ | --------------------------- | ------------------------------------------------------------ | ------------------------ |
| Update after | one epoch                   | one pattern                                                  | $mb$ patterns            |
| Gradient     | most accurate               | noisy                                                        | in between               |
| Speed        | slower (waits for all data) | can be faster                                                | good                     |
| $\eta$       | higher                      | **smaller**                                                  | tune it                  |
| Parallelism  | yes                         | no                                                           | yes (fits GPU memory)    |
| Extra        | smoothest                   | can escape local minima, regularizing noise, may be unstable | scales to huge data sets |

**DO**

- **(!)** **shuffle** the patterns (random order in each epoch) for on-line/mini-batch, otherwise a fixed order biases the descent;
- use random sampling to build the mini-batches (unbiased estimate).

**DON'T**

- compare batch and on-line at the same $\eta$ - see below.

$\to$ [[Batch Online and Mini-Batch]]

---

## 3. Learning rate $\eta$

**DO**

- **(!)** divide the gradient by $l$ (Least Mean Squares): it makes $\eta$ independent of the number of data, and then $0<\eta<1$;
- typical search range in LMS: $[0.01,\ 0.5]$ - but always find it by [[Grid Search]];
- consider decaying it linearly until step $\tau$ and then keeping it fixed and small ($\eta_\tau\approx1\%$ of $\eta_0$, $\tau$ a few hundred steps) - especially with mini-batch, where the gradient does not go to zero near a minimum;
- adaptive rates (AdaGrad, RMSProp, Adam) exist and are popular for DL.

**DON'T**

- **use the same $\eta$ for batch and on-line**: with LMS, on-line needs a much smaller $\eta$, by roughly the factor $1/l$ used to divide the batch gradient. For mini-batch, divide by $mb$ and retune, or just divide $\eta$ by $l$ too;
- pick $\eta$ too high (instability, divergence) or too low (stuck) - **look at the [[Learning Curve]]**.

$\to$ [[Learning Rate]]

---

## 4. Momentum $\alpha$

**DO**

- **(!)** use it: $\Delta w_{new}=-\eta\,\partial E/\partial w+\alpha\,\Delta w_{old}$, with $0<\alpha<1$, typically $[0.5,\ 0.9]$;
- remember it allows a **higher** $\eta$ (inertia), it is faster on plateaus and damps oscillations;
- keep $\eta$, $\alpha$ and $\lambda$ **independent** of each other, so their effects can be controlled separately (see the combined update below).

**DON'T**

- assume it always helps - it was born for batch mode; in on-line/mini-batch it becomes a moving average over different examples, which is a different thing. Check whether it is really needed, like any heuristic.
- expect Nesterov momentum to help in the stochastic case (it improves convergence rate for **batch** mode).

$\to$ [[Momentum]]

---

## 5. Regularization $\lambda$

**DO**

- **(!)** use it - it is the main tool for complexity control;
- $\operatorname{Loss}(w)=\sum_p(d_p-o(x_p))^2+\lambda\lVert w\rVert^2$, effect = weight decay, add $2\lambda w$ to the gradient;
- generally very low values, e.g. $0.01$ - **selected by model selection**;
- leave the bias $w_0$ out of the penalty (or give it its own coefficient);
- for on-line/mini-batch, compare fairly with batch using $\lambda\times(mb/l)$ - or just let model selection pick different $\lambda$ for each;
- keep **Loss** (data + penalty, minimized) distinct from **Error** (data term, reported).

**The combined update, keeping the three hyperparameters independent:**

$$
\Delta w_{tu}=\eta\,\delta_t o_u+\alpha\,\Delta w_{old,tu},
\qquad
w_{new,tu}=w_{tu}+\Delta w_{tu}-\lambda w_{tu}
$$

(and divide by $l$ only the gradient of $E$, where the sum over the patterns is).

**DON'T**

- expect regularization to fix an unstable **learning curve** - it controls complexity, not convergence stability;
- confuse it with early stopping: ES is empirical and spends validation data; regularization is principled and lets the VL curve follow the TR curve. You can use both.

$\to$ [[Regularization]]

---

## 6. Number of hidden units

**DO**

- treat it as a **model selection** issue, together with $\lambda$ (cross-validation on a validation set);
- remember it relates to the input dimension and the size of the TR set;
- allow **more** units if you use appropriate regularization;
- consider constructive approaches ([[Cascade Correlation]]) or pruning if you want the algorithm to decide.

**DON'T**

- too few $\to$ [[Underfitting]]; too many with no regularization $\to$ [[Overfitting]].

---

## 7. Stopping

**DO**

- basic criterion: mean error $<\epsilon$, if you know the tolerance of the data; or max error; or number of misclassified for classification;
- internal criteria: near-zero gradient, no relevant weight change, no significant error decrease (e.g. $<0.1\%$) - applied over $k$ epochs (**patience**), otherwise they fire prematurely;
- **(!)** always stop after an excessive number of epochs, just to escape a too slow convergence;
- [[Early Stopping]] with a **validation** set, with patience.

**DON'T**

- **stop at an arbitrary fixed number of epochs**: too small $\to$ underfitting (and it unfairly favours a high $\eta$, which needs fewer epochs), too large $\to$ overfitting or wasted time, and one number cannot fit every configuration nor every fold of a CV - and what would you use for the final retraining?
- stop on the **test** error - that turns the test set into a validation set;
- aim at the global minimum of $R_{emp}$ - it is likely an overfitting solution.

$\to$ [[Stopping Criteria]], [[Early Stopping]]

---

## 8. Multiple minima / trials

**DO**

- **(!)** try several random starting configurations (5-10 or more);
- report **mean $\pm$ variance** of the error over the trials;
- if you need one final model: best on VL, or median, or random (or a committee) - and **state which**.

**DON'T**

- worry about the global minimum: a good local minimum is normally enough, and we are minimizing $R$ anyway, not $R_{emp}$;
- ignore high variance - it is bad in itself.

$\to$ [[Multiple Minima]]

---

## 9. Input / output representation

**DO**

- **(!)** 1-of-k (one-hot) encoding for symbolic inputs - a frequent fault;
- standardize ($v-\text{mean}$)/std, or rescale to $[0,1]$, when the scales differ;
- **regression:** linear output unit(s) **(!)**;
- **classification:** 1 unit (binary) or 1-of-K multi-output; sigmoid + threshold, or softmax + cross-entropy; tanh often learns faster;
- use $0.9$ instead of $1$ as target (and $-0.9$ for $-1$) to avoid asymptotic convergence;
- keep the target range inside the output range of the units.

**DON'T**

- encode "missing" as $0$ if $0$ is a legal input value.

$\to$ [[Data and Preprocessing]], [[Activation Functions]]

---

## 10. How to search the hyperparameters

**DO**

- **(!)** [[Grid Search]] over **all the combinations** of the significant hyperparameters;
- coarse/wide grid first (exponentially growing values) to find the good regions, then finer grids in the smaller range;
- select on a validation set or an inner [[Cross Validation]] loop;
- fix some values in a preliminary phase only after checking they are not relevant (stability of training, variance of the results);
- [[Random Search]] when the space gets big - the budget is independent of the number of hyperparameters.

**DON'T**

- **select sequentially** - the order introduces a bias (an $\eta$ chosen with 10 units reused for 100 units), and hyperparameters interact;
- use the **test set** for tuning, in any form;
- throw an expensive automatic search at the problem without thinking - grid search teaches you more, and time is finite.

$\to$ [[Hyperparameter Search]], [[Validation Pitfalls]]

---

## 11. Sanity-checking an implementation

Small implementation errors are hard to spot, because a NN often still "works" while hampered.

**DO**

- test on a **small, easy, artificial benchmark** first, where a tiny network is known to reach very high accuracy quickly - if you can't get state-of-the-art results there with few units, the code is wrong;
- judge the **shape** of the [[Learning Curve]], not the absolute numbers: are you close to a good convergence or far from it? All the MSE curves should come out smooth.

**DON'T**

- conclude the implementation is correct because a benchmark passes: good results do **not** guarantee correctness, while bad results **do** require revising the code and the settings.

---

## Quick sanity checklist

1. **(!)** Look at the [[Learning Curve]]. Smooth and decreasing? If it diverges, $\eta$ too high; if it is flat, $\eta$ too low; if it is irregular, batch too small or $\eta$ too high.
2. Is the gradient divided by $l$? Is $\eta$ consistent with the batch mode you chose?
3. Are the weights small and random?
4. Are symbolic inputs one-hot encoded?
5. Are inputs standardized?
6. Several trials, mean $\pm$ std reported?
7. Was the test set separated **before** anything else?
8. Is the grid over **all** the significant hyperparameters together?
9. Are $\eta$, $\alpha$ and $\lambda$ independent in your update rule?
10. Did any test result influence a design choice? If yes, it is no longer a test.

---

## Which hyperparameters actually matter

The ones connected to the **VC-dim**, i.e. to the capacity: number of hidden units, $\lambda$, number of epochs, degree - and for [[SVM]], $C$ and the kernel parameters. The rest ($\eta$, $\alpha$, batch size, initialization, choice of [[Optimizers|optimizer]]) affect *whether and how fast you get there*, not the complexity of what you get.

Worth doing: relate the optimization of the loss with a penalty term to the VC-bound, to understand the role of #units and $\lambda$ in the under/overfitting cases.

## Related

- [[Training]], [[Neural Networks]], [[Validation]]
