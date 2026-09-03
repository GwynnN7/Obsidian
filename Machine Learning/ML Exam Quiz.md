---
tags: [machine-learning, exam, practice]
---

# ML Exam Quiz

Rapid-recall drilling: **105 items** - 62 True/False and 43 multiple choice - over the whole course. The companion open-question/derivation sheet is [[ML Exam Questions]].

Most of the False statements and most of the wrong options are **the misconceptions the notes explicitly warn about**, not random noise. If one of them looks true to you, that is the note to reread.

> [!tip] How to use it
> T/F: answer a whole block of eight, then open the key. Aim for 8/8 - these are meant to be gettable, and a miss is a real gap. MCQ: exactly one option is correct; before opening the answer, say *why each other option is wrong*. That is the part the oral exam actually tests.

Index: [[ML Essentials]] · [[Machine Learning]] · [[Hyperparameter Guidelines]]

---

# Part 1 - True or False

## Block A - Frame and inductive bias

1. The empirical risk on the training set is an unbiased estimate of the true risk of the hypothesis returned by ERM.
2. A learner with no inductive bias can still generalize, provided the training set is large enough.
3. "Linear model" refers to the shape of the function in the input space.
4. Restricting $H$ to conjunctions of literals cuts the number of semantically distinct hypotheses from $2^{2^n}$ to $3^n+1$.
5. In practice ML prefers a strong **language** bias over a search bias, because a smaller $H$ is easier to search.
6. The version space is the subset of $H$ consistent with every training example.
7. When a penalty term is present, the "Loss" minimized during training and the "Error" reported in tables are the same number.
8. An inductive learner with a bias is equivalent to a deductive system in which the bias is written down as an explicit premise.

> [!success]- Answers A
> 1. **F** - it is optimistically biased: the same data chose the hypothesis. Training error is not a risk estimate. $\to$ [[Generalization and Risk]]
> 2. **F** - proved false in [[Concept Learning]]: for any unobserved $x_i$ the version space splits in half, so the answer is "don't know". Only seeing *every* instance helps, i.e. a lookup table.
> 3. **F** - linear in the **parameters** $\boldsymbol{w}$. That is exactly why [[Linear Basis Expansion]] (curved in $\boldsymbol{x}$) is still a linear model.
> 4. **T** - three choices per variable ($l_i$, $\lnot l_i$, don't-care), $+1$ for *false*. For $n=4$: $65536\to82$.
> 5. **F** - the opposite. We prefer an **expressive $H$ + search bias**, so the unknown target is not excluded *a priori*; the bias moves into an incomplete search. $\to$ [[Hypothesis Space and Inductive Bias]]
> 6. **T** - $VS(H,D)=\{h\in H\mid h(x)=d(x)\ \forall(x,d)\in D\}$.
> 7. **F** - $\operatorname{Loss}=$ data error $+$ penalty; **Error/Risk** is the data term alone, and that is what you report. $\to$ [[Loss Functions]]
> 8. **T** - and the useful corollary: a learner with no bias has no premise to add, so it can prove nothing about unseen instances.

## Block B - Generalization

1. Bias and variance are properties computed across **different training sets**, not measurable from a single training run.
2. The noise term of the bias-variance decomposition can be reduced by choosing a better model class.
3. Increasing $\lambda$ increases the variance and decreases the bias.
4. On a plot indexed by $\lambda$, model complexity grows towards the right of the axis.
5. The Bayes optimal classifier has zero error rate.
6. Bagging mainly reduces variance; boosting mainly reduces bias.
7. As complexity grows, test-error curves obtained from different training sets fan out, and that spread *is* the variance term.
8. The classical U-shaped test-error curve is a statement about the number of parameters.

> [!success]- Answers B
> 1. **T** - there is a different $h$, and a different $y$, for every training set that could have been drawn. The darts picture is good but in ML you rarely see the individual throws.
> 2. **F** - it is $\sigma^2$, **irreducible** and model-independent: even the optimal solution can be wrong. It is the floor set by the [[Bayes Optimal Classifier]].
> 3. **F** - reversed. High $\lambda$ $\to$ rigid model $\to$ **high bias, low variance** ([[Underfitting]]).
> 4. **F** - complexity grows to the **left**, since low $\lambda$ means a complex model. Read the axis before reading the curves.
> 5. **F** - its error rate *is* the **Bayes rate**, the minimum achievable given the distribution; it is zero only for noiseless, non-overlapping classes.
> 6. **T** - averaging over resampled sets attacks variance directly; boosting builds complexity incrementally from weak learners.
> 7. **T** - training curves instead fall monotonically and stay bunched. This is why a single split can be lucky or unlucky, hence [[Cross Validation]] and mean $\pm$ SD.
> 8. **F** - it is about **capacity**. Push the parameter count far past zero training error and the test error can fall again: **double descent**, and epoch-wise **grokking**. $\to$ [[Deep Learning]]

## Block C - Capacity and bounds

1. $VC(H)=p$ requires that **every** configuration of $p$ points be shattered by $H$.
2. The VC dimension of a model class always equals its number of free parameters.
3. The VC-confidence term can be computed before any training takes place.
4. k-NN has a finite VC dimension, roughly $l/k$.
5. Linear separators (LTU) in $\mathbb{R}^n$ have $VC=n+1$.
6. SRM requires the hypothesis spaces to be nested **and** ordered by increasing VC dimension - both conditions.
7. The VC bound is tight enough to be used as a practical estimate of the generalization error.
8. AIC and BIC can be applied to any model, deep networks included.

> [!success]- Answers C
> 1. **F** - **at least one** configuration of $p$ points must be shattered, and **no** configuration of $p+1$. Three collinear points defeat lines, and $VC$ is still 3.
> 2. **F** - related but not equal. $\operatorname{sign}(\boldsymbol{x}\cdot\boldsymbol{x}-b)$ and $\operatorname{sign}(k\boldsymbol{x}\cdot\boldsymbol{x}-b)$ differ in parameters but have $VC$ 1 and 2; there are one-parameter families with infinite VC. In a NN, capacity depends on the number of weights **and their size**.
> 3. **T** - $\epsilon$ needs only $VC$, $N$ and $\delta$. And the bound holds for the whole class at once, so you may search over it.
> 4. **F** - **infinite**. ($l/k$ is its *effective number of parameters*, a different notion.)
> 5. **T** - an upper bound: constraining the model can lower it, which is exactly what the margin does in [[SVM]].
> 6. **T** - the nesting is what makes "one step more complex" well defined; the VC ordering is what makes the bound monotone.
> 7. **F** - worst case and very conservative, sometimes hundreds of times larger than the real overfitting effect. Good for **ranking** models, useless for assessment.
> 8. **F** - they need a likelihood and are limited to models **linear in the parameters**. $\to$ [[Statistical Learning Theory]]

## Block D - Linear models, SVM and kernels

1. On a linearly separable set, LMS is guaranteed to end with zero misclassifications.
2. The perceptron convergence bound $q\le(R/\gamma)^2$ depends on the number of patterns and on the input dimensionality.
3. $(\boldsymbol{X}^T\boldsymbol{X}+\lambda\boldsymbol{I})$ is invertible for every $\lambda>0$, whatever $\boldsymbol{X}$.
4. In a trained SVM the support vectors are always a small fraction of the training set.
5. The soft-margin dual objective is identical to the hard-margin one; only the constraint on $\alpha_i$ changes.
6. $b_o$ may be recovered from any support vector, including those with $\alpha_i=C$.
7. A gaussian RBF kernel corresponds to a feature space of infinite dimension.
8. Because complexity is ruled by the margin and not by the dimension, SVM solves the curse of dimensionality.

> [!success]- Answers D
> 1. **F** - LMS takes the gradient of the squared error computed *before* the threshold, so its delta $(d-\boldsymbol{w}^T\boldsymbol{x})$ is nonzero even on correctly classified points: it can misclassify on a separable problem. The [[Perceptron]] rule cannot.
> 2. **F** - neither appears. $q\le(R/\gamma)^2$ is (data radius / gap width)$^2$: **pure geometry**.
> 3. **T** - eigenvalues $\mu_i+\lambda>0$ since $\boldsymbol{X}^T\boldsymbol{X}$ is PSD. Ridge also repairs the degenerate least-squares case.
> 4. **F** - assumed low in the discussion, not always true: ≈42% of the data in the classic mixture example with a tuned $C$, ≈45% with an RBF kernel.
> 5. **T** - the slacks cancel out and leave a single trace, the box $0\le\alpha_i\le C$.
> 6. **F** - only points with $0<\alpha_j<C$. A point with $\alpha_j=C$ may have $\xi_j>0$, so its constraint is not tight and it yields the wrong $b$.
> 7. **T** - and with $\sigma\to0$ it behaves like 1-NN, hence infinite VC unless regularized. Small $\sigma$ therefore demands a **lower** $C$, and the two must be searched **together**.
> 8. **F** - it handles a high-dimensional **feature** space well; the curse lives in the **input** space and is untouched. $\to$ [[Curse of Dimensionality]]

## Block E - Neural networks and training

1. Backpropagation is a type of neural network.
2. An MLP whose hidden units are linear has the same hypothesis space as a single-layer linear model.
3. Backpropagation costs $O(\#W^2)$, with $\#W$ the number of weights.
4. The universal approximation theorem tells you how many hidden units are needed for a given accuracy.
5. Regularization is a technique for stabilizing the convergence of training.
6. Weight decay is exactly the gradient contribution of the L2 penalty term.
7. Choosing the stopping epoch on the test error is acceptable as long as the training error is reported alongside.
8. Comparing batch and on-line training at the same learning rate $\eta$ is a fair comparison.

> [!success]- Answers E
> 1. **F** - it is a **training algorithm**. Never say "backprop NN".
> 2. **T** - the one true statement in this block. A composition of linear maps is linear, so the network collapses to $\tilde{\boldsymbol{W}}\boldsymbol{x}$ and the depth is wasted. Everything depth buys requires a non-linear $f$.
> 3. **F** - $O(\#W)$: the same $\delta_t$, computed once, is reused for every weight of unit $t$, and output deltas are reused below. With $10^9$ weights that is $10^9$ vs $10^{18}$ operations.
> 4. **F** - an **existence** theorem. It gives neither the number of units nor the learning algorithm.
> 5. **F** - the classic misunderstanding. It controls **complexity** (the VC-dim, which in a NN depends on the number of weights *and their values*), not the stability of the [[Learning Curve]].
> 6. **T** - $\partial(\lambda\lVert\boldsymbol{w}\rVert^2)/\partial w=2\lambda w$, hence $w_{new}=w+\eta\Delta w-2\lambda w$. Keep $\eta$ off $\lambda$ so the two stay independent.
> 7. **F** - that turns the test set into a validation set and destroys the risk estimate. Early stopping is a **model-selection** decision.
> 8. **F** - dividing the gradient by $l$ or by $mb$ rescales the effective $\eta$. (Same for $\lambda$: use $\lambda\times(mb/l)$, or let model selection handle it.)

## Block F - Architectures and deep learning

1. Weight sharing in a CNN reduces the number of **connections** in the layer.
2. After $l$ message-passing steps, the state of a node in a DGN covers a context of radius $l$.
3. The echo state property is enforced by limiting the spectral radius of the reservoir matrix.
4. In an Echo State Network the recurrent weights are trained by BPTT, only faster.
5. Depth extends what a network can **represent** beyond the reach of a single hidden layer.
6. Parity over $N$ bits needs $\sim2^{N-1}$ gates at depth 2, but only $\sim3(N-1)$ with $\log N$ layers.
7. A CNN kernel can be applied directly to the neighbourhood of a graph vertex.
8. In a randomized neural network the readout is trained by pseudoinverse or ridge regression, in closed form.

> [!success]- Answers F
> 1. **F** - it reduces the number of **free parameters** while keeping the connections. That is what makes it *regularization by architecture*.
> 2. **T** - so depth is not a tuning choice: it is how information reaches distant nodes. One step sees only $\mathcal{N}(v)$.
> 3. **T** - contractivity of the state transition, so the state echoes only the input history and forgets the initial state.
> 4. **F** - they are **never trained**: random, sparse, fixed. Only the linear readout is fitted. That is the whole point (no BPTT cost).
> 5. **F** - by [[Universal Approximation]] one hidden layer already suffices in principle. Depth buys **efficiency** (no-flattening: exponentially fewer units) and often learnability.
> 6. **T** - DNF needs one AND per positive configuration ($2^{N-1}+1$); a binary tree of XOR nodes needs $N-1$ internal nodes at 3 gates each. For $N=8$: 129 vs 21.
> 7. **F** - a kernel needs a regular grid; graph neighbours are **unordered and variable in size**. The fix keeps local receptive fields + weight sharing and drops the grid, adding permutation-invariant aggregation.
> 8. **T** - the readout is a linear model on the random features: $\boldsymbol{W}^{out}=\boldsymbol{H}^{+}\boldsymbol{d}$, or $(\boldsymbol{H}^T\boldsymbol{H}+\lambda\boldsymbol{I})^{-1}\boldsymbol{H}^T\boldsymbol{d}$.

## Block G - Learning without a teacher

1. In the on-line K-means / VQ rule, only the winner prototype is moved.
2. K-means converges to the global minimum of its squared-error criterion.
3. In a SOM, $\lVert i-j\rVert$ inside the neighbourhood function is the distance between the two units' **weight vectors**.
4. Replacing the SOM neighbourhood function by $h_{i,j}=\delta_{ij}$ gives on-line K-means.
5. Without a target, the chosen metric constitutes the whole inductive bias.
6. In a SOM both $\eta(t)$ and $\sigma(t)$ grow with $t$.

> [!success]- Answers G
> 1. **T** - $\Delta\boldsymbol{w}_{i^*}=\eta(\boldsymbol{x}_i-\boldsymbol{w}_{i^*})$, the derivative of the VQ loss: the winner moves towards the pattern.
> 2. **F** - it is stochastic gradient descent on a non-convex criterion: local minima and dependence on the initialization. Hence multiple restarts.
> 3. **F** - it is the distance between their **coordinates on the lattice**, never between weight vectors. That is what makes the map topology-preserving.
> 4. **T** - the neighbourhood term *is* the SOM; remove it and only the winner learns.
> 5. **T** - and evaluating the result is largely subjective.
> 6. **F** - both **decrease** (ordering phase, then convergence phase).

## Block H - Validation

1. K-fold CV used for model selection does return a final model, after retraining on the whole data set.
2. Using K-fold CV is by itself enough to protect against subset selection bias.
3. Feature selection may be run on the entire data set before splitting, provided a test set is held out afterwards.
4. Leave-One-Out wastes almost no data, but is expensive and can have high variance.
5. On a data set with 99% positives, accuracy is still an adequate summary metric.
6. The diagonal of a ROC curve corresponds to random guessing.
7. Specificity equals $1-$ FP rate.
8. Hyperparameters may be tuned one at a time without introducing bias, as long as each is tuned on validation data.

> [!success]- Answers H
> 1. **T** - pick the best $\Lambda$ on the validation results, then retrain on all the data. Used for **assessment** instead, it gives one model per fold and what you keep is the estimate with its variability.
> 2. **F** - it does not solve it and can even be more convincing, hence more misleading. The fix is to leave samples out **before** selecting, or to redo the selection inside every fold.
> 3. **F** - one of the three classic mistakes (with scaling before splitting, and reporting VL as TS). In the 1000-random-features counterexample this is exactly what produces a fake 99% instead of the true 50%.
> 4. **T** - which is why 5- or 10-fold is the recommended compromise.
> 5. **F** - a trivial "always positive" classifier already scores 99%. Report recall/precision or the confusion matrix. (And on a *balanced* binary task, 50% is a coin.)
> 6. **T** - and better classifiers lie above it; the summary number is the AUC.
> 7. **T** - specificity $=TN/(TN+FP)$ and FP rate $=FP/(TN+FP)$.
> 8. **F** - sequential tuning introduces bias through the cross effects: an $\eta$ chosen with 10 hidden units need not suit 100. Search a **grid with all the combinations** of the significant hyperparameters (coarse to fine is fine, but keep them together).

---

# Part 2 - Multiple choice

*Exactly one option is correct. Before opening the key, be able to say why each of the other three fails.*

## Frame and inductive bias

**M1.** With $n$ binary attributes, the number of *semantically distinct* conjunctive hypotheses is
&nbsp;&nbsp;**a)** $2^{2^n}$ &nbsp;&nbsp;**b)** $3^n$ &nbsp;&nbsp;**c)** $3^n+1$ &nbsp;&nbsp;**d)** $2^n$

> [!success]- Answer M1
> **c)**. Three choices per variable - $l_i$, $\lnot l_i$, don't-care - plus one for the single *false* hypothesis (every $h$ containing $l_i\wedge\lnot l_i$ collapses to it), which is why **b** is wrong. **a** is the number of Boolean functions on $n$ inputs, i.e. the *unbiased* $H$. **d** is the number of input instances.

**M2.** Which of these is a **search** bias?
&nbsp;&nbsp;**a)** gradient descent started from small random weights &nbsp;&nbsp;**b)** $H$ restricted to conjunctions of literals &nbsp;&nbsp;**c)** polynomials of degree at most 3 &nbsp;&nbsp;**d)** $H$ restricted to linear functions

> [!success]- Answer M2
> **a)**. Language bias restricts *what can be written* (**d**, **b**, **c**); search bias restricts *how the space is explored* - the starting point, the update rule, where you stop. Note that a and d are the same kind of restriction stated twice.

**M3.** The inner loss used for density estimation is
&nbsp;&nbsp;**a)** $-\ln h(\boldsymbol{x}_p)$ &nbsp;&nbsp;**b)** the 0/1 loss &nbsp;&nbsp;**c)** $\lvert d-y\rvert-\epsilon$ &nbsp;&nbsp;**d)** $(d_p-h(\boldsymbol{x}_p))^2$

> [!success]- Answer M3
> **a)** - related to maximizing the log-likelihood. **d** is regression, **b** classification, **c** the $\epsilon$-insensitive loss of [[SVM Regression]]. Clustering/VQ would be $\lVert\boldsymbol{x}_p-h(\boldsymbol{x}_p)\rVert^2$.

**M4.** Empirical Risk Minimization is best described as
&nbsp;&nbsp;**a)** a validation schema &nbsp;&nbsp;**b)** a theorem guaranteeing $R_{emp}\to R$ &nbsp;&nbsp;**c)** an inductive principle &nbsp;&nbsp;**d)** a regularization technique

> [!success]- Answer M4
> **c)**. Replacing the unknowable integral $R$ by a finite average is an assumption, not a result. Whether it works depends on the amount of data and the capacity of $H$ - the question [[Statistical Learning Theory]] answers, and there the convergence needs conditions, so **b** is wrong as stated.

**M5.** Which quantity is the one you report in tables and plots?
&nbsp;&nbsp;**a)** data error $+$ penalty &nbsp;&nbsp;**b)** the VC-confidence &nbsp;&nbsp;**c)** the penalty alone &nbsp;&nbsp;**d)** the data term alone

> [!success]- Answer M5
> **d)** - the **Error/Risk**. **a** is the **Loss**, the training objective; mixing the two makes a regularized model look worse than it is and makes runs with different $\lambda$ incomparable.

## Generalization

**M6.** A model that is too rigid for the task has
&nbsp;&nbsp;**a)** both high &nbsp;&nbsp;**b)** high bias, low variance &nbsp;&nbsp;**c)** high variance, low bias &nbsp;&nbsp;**d)** both low

> [!success]- Answer M6
> **b)** - [[Underfitting]]. **c** is the flexible/overfitting case. **a** happens with a badly chosen model, but is not what "rigid" means; **d** would be the ideal, unreachable in general.

**M7.** Which term of the decomposition does **not** depend on the model?
&nbsp;&nbsp;**a)** noise &nbsp;&nbsp;**b)** all three do &nbsp;&nbsp;**c)** bias$^2$ &nbsp;&nbsp;**d)** variance

> [!success]- Answer M7
> **a)** - $\sigma^2$ is irreducible: the same $\boldsymbol{x}$ may carry different targets. It sets the error floor, which is the same statement as the Bayes rate.

**M8.** Averaging models trained on bootstrap resamples attacks mainly
&nbsp;&nbsp;**a)** the variance &nbsp;&nbsp;**b)** the VC dimension &nbsp;&nbsp;**c)** the noise &nbsp;&nbsp;**d)** the bias

> [!success]- Answer M8
> **a)** - [[Bagging]]. The picture is the "low $\lambda$" case: individual fits are wild, but $\bar h(\boldsymbol{x})$ is close to $f$. **d** is [[Boosting]]. Ensembles help only if the members are **diversified**.

**M9.** The Bayes rate is
&nbsp;&nbsp;**a)** always zero &nbsp;&nbsp;**b)** the minimum achievable error given the data distribution &nbsp;&nbsp;**c)** the error of the best hypothesis in $H$ &nbsp;&nbsp;**d)** the training error of the optimal classifier

> [!success]- Answer M9
> **b)**. **c** is a different (larger) quantity - it depends on the chosen $H$, whereas the Bayes rate does not. k-NN approximates the Bayes rule by replacing the conditional probability at a point with the one in a neighbourhood, estimated by sample proportions.

**M10.** No Free Lunch says that
&nbsp;&nbsp;**a)** complexity control is unnecessary &nbsp;&nbsp;**b)** all models perform equally in practice &nbsp;&nbsp;**c)** flexible models always beat rigid ones &nbsp;&nbsp;**d)** no learner is universally best: an advantage on some problems is paid for on others

> [!success]- Answer M10
> **d)**. **b** overreads it: models still differ in **flexibility** and in **controllability of complexity**, and real problems are not drawn uniformly from all possible functions. **c** contradicts the theorem; **a** contradicts everything else in the course.

## Capacity and bounds

**M11.** $VC(\text{lines in }\mathbb{R}^2)=3$ because
&nbsp;&nbsp;**a)** $VC=n+1$ holds for every model &nbsp;&nbsp;**b)** a line has 3 parameters &nbsp;&nbsp;**c)** every set of 3 points is shattered &nbsp;&nbsp;**d)** some set of 3 points is shattered and no set of 4 is

> [!success]- Answer M11
> **d)** - the asymmetric definition. **c** is false: 3 collinear points labelled $+,-,+$ cannot be separated, and that does not matter. **b** confuses VC with the parameter count (see M12). **a** is true only for linear separators, and even there as an upper bound.

**M12.** Comparing two model classes with the same number of free parameters, their VC dimensions
&nbsp;&nbsp;**a)** must differ &nbsp;&nbsp;**b)** may differ &nbsp;&nbsp;**c)** are both infinite &nbsp;&nbsp;**d)** must be equal

> [!success]- Answer M12
> **b)**. Circles centred at the origin: $\operatorname{sign}(\boldsymbol{x}\cdot\boldsymbol{x}-b)$ has $VC=1$ while $\operatorname{sign}(k\boldsymbol{x}\cdot\boldsymbol{x}-b)$ has $VC=2$; redundant parameters can always be added; there exist one-parameter classes with infinite VC. For a NN, capacity depends on the number of units **and on the magnitude of the weights** - which is why weight decay reduces effective capacity without removing a single parameter.

**M13.** In the VC bound $R\le R_{emp}+\epsilon$, the term $\epsilon$
&nbsp;&nbsp;**a)** shrinks as $N$ grows and grows with the VC dimension &nbsp;&nbsp;**b)** does not depend on the confidence $\delta$ &nbsp;&nbsp;**c)** grows with $N$ &nbsp;&nbsp;**d)** shrinks as the VC dimension grows

> [!success]- Answer M13
> **a)**. Explicitly $\epsilon=\sqrt{\big(VC(\ln\frac{2N}{VC}+1)-\ln\frac{\delta}{4}\big)/N}$: $N$ sits in the denominator, $VC$ in the numerator, and $\delta$ is there too, so **b** is false. The condition $VC<N$ is the formal version of "you need more data than capacity".

**M14.** Hard-margin SVM implements SRM by
&nbsp;&nbsp;**a)** fixing the complexity and minimizing the training error &nbsp;&nbsp;**b)** fixing the training error at zero and minimizing the capacity term &nbsp;&nbsp;**c)** cross-validating $C$ &nbsp;&nbsp;**d)** adding hidden units until the bound stops falling

> [!success]- Answer M14
> **b)** - "approach 2": separability makes $R_{emp}=0$ by construction, and maximizing the margin minimizes $VC\le\min(\lceil D^2/\rho^2\rceil,m_0)+1$. **a** is what we do with NN. **c** describes the *soft*-margin case, where the automatic SRM is precisely what we lose.

**M15.** Which criterion measures complexity as a **description length**?
&nbsp;&nbsp;**a)** BIC &nbsp;&nbsp;**b)** Tikhonov &nbsp;&nbsp;**c)** AIC &nbsp;&nbsp;**d)** MDL

> [!success]- Answer M15
> **d)** - bits for the model **plus** bits for the data given the model, so a perfectly fitting but huge model costs as much as a small one with many residuals. AIC/BIC count parameters ($-2\log\hat L+2k$ and $-2\log\hat L+k\ln N$), SRM counts VC, Tikhonov measures $\lVert\boldsymbol{w}\rVert$. They can disagree - that is the interesting part.

**M16.** Cover's theorem holds *provided that*
&nbsp;&nbsp;**a)** the space is **not** densely populated &nbsp;&nbsp;**b)** the mapping is linear &nbsp;&nbsp;**c)** the input dimension is at most 3 &nbsp;&nbsp;**d)** the feature space is densely populated

> [!success]- Answer M16
> **a)**. Ignore that clause and you have simply re-described the [[Curse of Dimensionality]]. **b** is self-defeating: a linear map cannot create separability. The theorem justifies fixed, adaptive, random and kernel-implicit expansions alike.

## Linear models, SVM and kernels

**M17.** If $\boldsymbol{X}^T\boldsymbol{X}$ is singular, the least-squares problem
&nbsp;&nbsp;**a)** has a unique solution &nbsp;&nbsp;**b)** can only be solved by gradient descent &nbsp;&nbsp;**c)** has infinitely many; take the minimum-norm one via SVD &nbsp;&nbsp;**d)** has no solution

> [!success]- Answer M17
> **c)** - $\boldsymbol{X}^{+}=\boldsymbol{V}\boldsymbol{\Sigma}^{+}\boldsymbol{U}^T$ gives the minimal-norm least-squares solution directly. **d** is wrong: the objective is convex, a minimum always exists. Adding a ridge term is the other standard fix, and it restores uniqueness.

**M18.** In $q\le(R/\gamma)^2$, the quantity $\gamma$ is
&nbsp;&nbsp;**a)** $\lVert\boldsymbol{w}^*\rVert/\alpha$ &nbsp;&nbsp;**b)** $\alpha$ &nbsp;&nbsp;**c)** $\alpha/\lVert\boldsymbol{w}^*\rVert$ &nbsp;&nbsp;**d)** $\max_i\lVert\boldsymbol{x}_i\rVert^2$

> [!success]- Answer M18
> **c)** - the **geometric** distance from the closest pattern to the hyperplane. $\alpha$ alone (**b**) is not scale-invariant: rescaling $\boldsymbol{w}^*$ doubles it while leaving the boundary unchanged, which is exactly why the bound divides by $\lVert\boldsymbol{w}^*\rVert^2$. **d** is $\beta=R^2$.

**M19.** The SVM margin in canonical form is
&nbsp;&nbsp;**a)** $\lVert\boldsymbol{w}\rVert/2$ &nbsp;&nbsp;**b)** $2\lVert\boldsymbol{w}\rVert$ &nbsp;&nbsp;**c)** $1/\lVert\boldsymbol{w}\rVert$ &nbsp;&nbsp;**d)** $2/\lVert\boldsymbol{w}\rVert$

> [!success]- Answer M19
> **d)**. **c** is the distance from the hyperplane to *one* support vector; the margin counts both sides. Maximizing it is minimizing $\tfrac12\lVert\boldsymbol{w}\rVert^2$ - same argmin, differentiable, and a convex QP.

**M20.** The Kuhn-Tucker complementarity condition implies that
&nbsp;&nbsp;**a)** $\alpha_i=0$ for every non-support vector, so the solution is sparse &nbsp;&nbsp;**b)** $\alpha_i=C$ for all $i$ &nbsp;&nbsp;**c)** all $\alpha_i>0$ &nbsp;&nbsp;**d)** $\boldsymbol{w}=\boldsymbol{0}$

> [!success]- Answer M20
> **a)**. From $\alpha_i(d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)-1)=0$: either $\alpha_i=0$ or the constraint is tight. Hence $\boldsymbol{w}_o=\sum_{i}^{N_s}\alpha_{o,i}d_i\boldsymbol{x}_i$ depends only on the points at the boundary - nice, because far points are irrelevant; less nice, because it hopes the boundary points are not the noisiest.

**M21.** An SVM with a gaussian kernel of very small width $\sigma$ behaves like
&nbsp;&nbsp;**a)** a linear SVM &nbsp;&nbsp;**b)** 1-NN &nbsp;&nbsp;**c)** a perceptron &nbsp;&nbsp;**d)** ridge regression

> [!success]- Answer M21
> **b)** - the kernel is so peaked that the reply for $\boldsymbol{x}_i$ is essentially just $d_i$: only the closest support vector contributes. Consequence: infinite VC unless regularized, so a small $\sigma$ requires a **lower** $C$, and the two must be searched jointly. A large $\sigma$ gives the opposite - a global average, low VC.

**M22.** Which is **not** a way to certify that a function is a valid kernel?
&nbsp;&nbsp;**a)** its Gram matrix is symmetric PSD on every finite sample &nbsp;&nbsp;**b)** it is the sum of two kernels &nbsp;&nbsp;**c)** it is monotone decreasing in $\lVert\boldsymbol{x}-\boldsymbol{y}\rVert$ &nbsp;&nbsp;**d)** it is the product of two kernels

> [!success]- Answer M22
> **c)** - "looks like a similarity" is not a criterion. **a** is Mercer's condition itself; **b** and **d** are closure properties (with $\alpha k_1$, $\alpha>0$), the cheapest route to a new kernel without touching Mercer. Note the two-layer perceptron kernel $\tanh(\beta_0\boldsymbol{x}^T\boldsymbol{x}_i+\beta_1)$ is valid only for some $\beta_0,\beta_1$.

## Neural networks and training

**M23.** The delta of a **hidden** unit $j$ is
&nbsp;&nbsp;**a)** $\big(\sum_k\delta_kw_{kj}\big)f_j'(net_j)$ &nbsp;&nbsp;**b)** $(d_j-o_j)f_j'(net_j)$ &nbsp;&nbsp;**c)** $\delta_ko_j$ &nbsp;&nbsp;**d)** $\eta\,\delta_j o_j$

> [!success]- Answer M23
> **a)** - the sum runs over every unit $k$ that $j$ feeds, because $E_p$ depends on $o_j$ only through them. **b** is the *output*-unit delta (and a hidden unit has no target $d_j$ at all). **c** is the weight *update* $\Delta_pw_{kj}$, **d** that update times the learning rate.

**M24.** Backpropagation's efficiency comes from
&nbsp;&nbsp;**a)** avoiding a matrix inversion &nbsp;&nbsp;**b)** the convexity of the loss &nbsp;&nbsp;**c)** parallel hardware &nbsp;&nbsp;**d)** computing each $\delta_t$ once and reusing it for all the weights of unit $t$

> [!success]- Answer M24
> **d)** - the "magic factorization": $O(\#W)$ instead of $O(\#W^2)$, and the output deltas are reused by the layer below. **b** is doubly wrong: the loss of an MLP is **not** convex ([[Multiple Minima]]).

**M25.** Weights are initialized small and random because
&nbsp;&nbsp;**a)** it enforces sparsity &nbsp;&nbsp;**b)** small numbers are cheaper to store &nbsp;&nbsp;**c)** it places units in the near-linear zone where $f'$ is large, and breaks symmetry &nbsp;&nbsp;**d)** it guarantees reaching the global minimum

> [!success]- Answer M25
> **c)**. $f'$ gates every update: in the saturated tails the deltas are tiny and learning stalls. Related practical rule: keep targets inside the output range (0.9 rather than 1), so the unit does not chase an asymptote. **d** is impossible - the surface is non-convex, and different inits give genuinely different models.

**M26.** Softmax outputs are naturally paired with
&nbsp;&nbsp;**a)** the $\epsilon$-insensitive loss &nbsp;&nbsp;**b)** MSE &nbsp;&nbsp;**c)** cross-entropy &nbsp;&nbsp;**d)** the 0/1 loss

> [!success]- Answer M26
> **c)**. Softmax outputs are positive and sum to 1, so they read as $p(\text{class}=k\mid\boldsymbol{x})$, and the natural loss on a probability is $-\sum_i\{d_i\log o_i+(1-d_i)\log(1-o_i)\}$. **d** is not differentiable, which is why it is never a *training* loss.

**M27.** A typical value for the momentum coefficient $\alpha$ is
&nbsp;&nbsp;**a)** $0.001$ &nbsp;&nbsp;**b)** $0.5-0.9$ &nbsp;&nbsp;**c)** $1.5$ &nbsp;&nbsp;**d)** $10$

> [!success]- Answer M27
> **b)**, with $0<\alpha<1$ required for stability - so **c** and **d** are not merely unusual but divergent. Effects: faster on plateaus, damping of oscillations in a badly conditioned canyon, and enough inertia to permit a larger $\eta$.

**M28.** Which statement about mini-batch training is correct?
&nbsp;&nbsp;**a)** the cost of one update grows with $l$ &nbsp;&nbsp;**b)** it gives a less noisy gradient than batch &nbsp;&nbsp;**c)** the cost of one update depends on $mb$, not on $l$ &nbsp;&nbsp;**d)** shuffling is unnecessary

> [!success]- Answer M28
> **c)** - which is exactly what extends learning to millions of examples, and to streams with no epochs at all. **d** is a real error: a fixed presentation order biases the descent. **b** is backwards - batch has the most accurate gradient estimate; mini-batch trades accuracy for updates per second.

## Architectures and deep learning

**M29.** A $28\times28$ input, kernel $3\times3$, padding $0$, stride $1$. The feature map is
&nbsp;&nbsp;**a)** $26\times26$ &nbsp;&nbsp;**b)** $28\times28$ &nbsp;&nbsp;**c)** $25\times25$ &nbsp;&nbsp;**d)** $27\times27$

> [!success]- Answer M29
> **a)**. $o=\lfloor(i-k+2p)/s\rfloor+1=\lfloor(28-3+0)/1\rfloor+1=25+1=26$. You would need $p=1$ ("same" padding) to keep **b**.

**M30.** Pooling provides
&nbsp;&nbsp;**a)** the layer's non-linearity &nbsp;&nbsp;**b)** more free parameters &nbsp;&nbsp;**c)** exact translation invariance &nbsp;&nbsp;**d)** approximate invariance to small translations

> [!success]- Answer M30
> **d)** - a max or a mean changes little if a few inputs shift, so a high output stays high for a slightly translated pattern. Pooling has **no** parameters, and the non-linearity comes from the units' activation function.

**M31.** In an RNN, the assumption that buys "one fixed-size model for sequences of any length" is
&nbsp;&nbsp;**a)** causality &nbsp;&nbsp;**b)** adaptivity &nbsp;&nbsp;**c)** contractivity &nbsp;&nbsp;**d)** stationarity

> [!success]- Answer M31
> **d)** - $\tau$ is the same at every $t$, i.e. weight sharing over time. **a** makes an internal state possible at all; **b** makes the encoding of the past *learnt* rather than designed; **c** is the ESN stability condition, a different matter.

**M32.** The architectural bias of a reservoir is that states are close in proportion to the length of the shared
&nbsp;&nbsp;**a)** subsequence &nbsp;&nbsp;**b)** suffix &nbsp;&nbsp;**c)** prefix &nbsp;&nbsp;**d)** alphabet

> [!success]- Answer M32
> **b)** - a **suffix**-based, Markovian organization of the state space, present even with untrained recurrent weights (and present with full training too). This is why Markovian tasks are easy for ESN, and why they are so much cheaper than BPTT.

**M33.** Permutation invariance in a Deep Graph Network comes from
&nbsp;&nbsp;**a)** global pooling alone &nbsp;&nbsp;**b)** the depth $L$ &nbsp;&nbsp;**c)** an order-blind aggregation operator plus weight sharing across nodes &nbsp;&nbsp;**d)** the adjacency matrix

> [!success]- Answer M33
> **c)**. $\mathrm{AGG}$ (sum, mean, max, attention) ignores the order of the neighbour set, and the same $\boldsymbol{W}^{(l)}$ is used at every node. **a** matters only for graph-level readouts; **b** buys *context radius*, not invariance.

**M34.** Depth in a neural network is best described as
&nbsp;&nbsp;**a)** an inductive bias: the target is a composition of simpler functions &nbsp;&nbsp;**b)** a guarantee against overfitting &nbsp;&nbsp;**c)** a form of data augmentation &nbsp;&nbsp;**d)** a way to represent functions a shallow net cannot represent

> [!success]- Answer M34
> **a)**. Expressivity is already settled by [[Universal Approximation]] (so **d** is wrong); depth buys **efficiency** - no-flattening results, the parity example - and it does so by assuming compositionality. When that assumption fits, an exponential saving; when it does not, [[No Free Lunch]] still applies, which also kills **b**.

## Unsupervised learning

**M35.** The on-line K-means update is
&nbsp;&nbsp;**a)** $\eta(\boldsymbol{w}_{i^*}-\boldsymbol{x}_i)$ &nbsp;&nbsp;**b)** $\eta(\boldsymbol{x}_i-\boldsymbol{w}_{i^*})$ for the winner only &nbsp;&nbsp;**c)** $\eta\,\boldsymbol{w}_{i^*}$ &nbsp;&nbsp;**d)** $\eta(\boldsymbol{x}_i-\boldsymbol{w}_j)$ for every $j$

> [!success]- Answer M35
> **b)** - the gradient of the VQ loss, in which only the winner's term is active. **a** has the sign reversed (the prototype would flee the data). **d** is what a SOM does *with a neighbourhood weight*, and without one it is simply wrong.

**M36.** In a SOM, decreasing $\sigma(t)$ over time serves to
&nbsp;&nbsp;**a)** increase the number of units &nbsp;&nbsp;**b)** speed up the winner search &nbsp;&nbsp;**c)** move from a global ordering phase to a local convergence phase &nbsp;&nbsp;**d)** normalize the inputs

> [!success]- Answer M36
> **c)** - a wide neighbourhood first organizes the map globally, then a narrowing one refines each unit locally. $\eta(t)$ decreases for the same reason. With $h_{i,j}=\delta_{ij}$ from the start you lose the ordering and get plain on-line K-means.

## Validation

**M37.** Which activity returns an *estimation value* rather than a model?
&nbsp;&nbsp;**a)** grid search &nbsp;&nbsp;**b)** training &nbsp;&nbsp;**c)** model assessment &nbsp;&nbsp;**d)** model selection

> [!success]- Answer M37
> **c)**. Selection (**d**, and **a** which is one of its implementations) returns a model; training returns fitted parameters. Confusing the two outputs is the root of most validation errors.

**M38.** The recommended compromise between bias and variance of the CV estimate is
&nbsp;&nbsp;**a)** LOO in all cases &nbsp;&nbsp;**b)** 5- or 10-fold &nbsp;&nbsp;**c)** 2-fold &nbsp;&nbsp;**d)** hold-out

> [!success]- Answer M38
> **b)**. **d** is cheap but split-dependent and unreliable; **a** wastes almost no data but is expensive and its estimate can have high variance. Add stratification, and repeat the CV when you are *comparing* models.

**M39.** In a nested (double) cross-validation, the **inner** loop is used for
&nbsp;&nbsp;**a)** model and hyperparameter selection &nbsp;&nbsp;**b)** training only &nbsp;&nbsp;**c)** assessment &nbsp;&nbsp;**d)** stratification

> [!success]- Answer M39
> **a)** - and the outer loop assesses the whole *selection procedure*, which is why a fresh selection must be run inside each outer fold. Skip that and you are back to subset selection bias.

**M40.** Precision is
&nbsp;&nbsp;**a)** $TN/(TN+FP)$ &nbsp;&nbsp;**b)** $TP/(TP+FP)$ &nbsp;&nbsp;**c)** $(TP+TN)/(TP+TN+FP+FN)$ &nbsp;&nbsp;**d)** $TP/(TP+FN)$

> [!success]- Answer M40
> **b)** - the share of *predicted* positives that really are positive. **d** is recall/sensitivity (share of *actual* positives found), **a** specificity, **c** accuracy. Learn them by which denominator you are conditioning on: a row of the confusion matrix (recall, specificity) or a column (precision).

**M41.** In the counterexample with 20-30 examples, 1000 random features and a random target, the honest test error is about
&nbsp;&nbsp;**a)** 0% &nbsp;&nbsp;**b)** 1% &nbsp;&nbsp;**c)** 99% &nbsp;&nbsp;**d)** 50%

> [!success]- Answer M41
> **d)** - a coin, because there is nothing to learn. The 99-100% observed on later splits is **subset selection bias**: the feature was chosen using all the data, so the test set had already been consumed. Fixes: separate the test set before *any* selection, or redo the feature selection inside every fold.

**M42.** The AUC of a classifier that guesses at random is
&nbsp;&nbsp;**a)** 0 &nbsp;&nbsp;**b)** undefined &nbsp;&nbsp;**c)** 1 &nbsp;&nbsp;**d)** 0.5

> [!success]- Answer M42
> **d)** - the area under the diagonal. **a** would mean systematically inverted rankings, which is a *perfect* classifier with the labels flipped, so it is not the worst case; **c** is perfect ranking.

**M43.** Early stopping is best classified as
&nbsp;&nbsp;**a)** a training heuristic that needs no extra data &nbsp;&nbsp;**b)** a model-selection decision requiring validation evidence &nbsp;&nbsp;**c)** something to be tuned on the test set &nbsp;&nbsp;**d)** equivalent to an L1 penalty

> [!success]- Answer M43
> **b)** - it is empirical complexity control, and it **spends** validation data (which is one of its costs versus a penalty in the loss). Inside CV it must be activated independently in every fold, and the cleanest way to retrain afterwards is to average the *training error* at each fold's validation optimum and refit to that level of fitting.

---

## Score sheet

| Part | Items | Your score |
|---|---|---|
| T/F A - Frame | 8 | |
| T/F B - Generalization | 8 | |
| T/F C - Capacity | 8 | |
| T/F D - Linear / SVM | 8 | |
| T/F E - NN / training | 8 | |
| T/F F - Architectures | 8 | |
| T/F G - Unsupervised | 6 | |
| T/F H - Validation | 8 | |
| MCQ M1-M43 | 43 | |
| **Total** | **105** | |

> [!important] If you missed one of these, reread the note, not the answer
> Anything in Block C or H, plus M14, M21 and M41: those are the items where the exam separates people who memorized a formula from people who understood what it is for.
