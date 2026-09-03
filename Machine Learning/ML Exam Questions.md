---
tags: [machine-learning, exam, practice]
---

# ML Exam Questions

Practice set for the written/oral exam, built from this vault. Mixed format: **[theory]** open questions to answer in words, **[derivation]** things you must be able to reproduce on paper, **[exercise]** short numerical items.

Answers are folded - try the question first, then click to open.

> [!tip] How to use it
> Do a whole Part with the answers closed, on paper. Only then open them. If an answer surprises you, go back to the linked note, not to this page.

Index: [[ML Essentials]] · [[Machine Learning]]

---

## Part A - The frame

### A1 [theory] Hypothesis space and inductive bias

Define *model*, *hypothesis*, *hypothesis space* and *inductive bias*. Distinguish **language bias** from **search bias** and give two examples of each. Which of the two does ML normally prefer, and why?

> [!success]- Answer
> - **Model:** the "language" that fixes which functions can be represented. **Hypothesis** $h$: one candidate approximation of the unknown target $f$. **Hypothesis space** $H$: all the $h$ the learner can output, e.g. $\{h(\boldsymbol{x},\boldsymbol{w})\}$ over all $\boldsymbol{w}$. **Inductive bias:** the set of assumptions that make generalization to unseen instances possible at all.
> - **Language bias** = restrictions on $H$ itself: only linear functions ([[Linear Models]]), only conjunctions of literals ([[Concept Learning]]).
> - **Search bias** = restrictions/preferences of the [[Learning Algorithm]] that explores $H$: least-squares ordered search, gradient descent from small initial weights, early stopping.
> - ML normally prefers an **expressive $H$ + search bias**: with a restrictive language the unknown target may be excluded *a priori* and can never be recovered. Concretely this means an *incomplete* search over a big space rather than a complete search over a small one.
> - Nice framing: an inductive learner with a bias is **equivalent to a deductive system** whose extra premise is the bias written out explicitly. A learner with no bias has nothing to add, so it can prove nothing about unseen instances.
>
> See [[Hypothesis Space and Inductive Bias]], [[Inductive Bias]].

### A2 [theory] Risk, empirical risk, ERM

Write the definitions of true risk and empirical risk. Why can we not compute $R$? What is ERM, and why is it an *inductive principle* rather than a theorem?

> [!success]- Answer
> $$R(h)=\mathbb{E}_Z[L(z,h)]=\int_Z L(z,h)\,p(z)\,dz,\qquad R_{emp}(h,D_r)=\frac1r\sum_{p=1}^{r}L(z_p,h)$$
> with $z=(\boldsymbol{x},d)$ drawn **i.i.d.** from an unknown $p(z)$.
> - $R$ is an integral against $p(z)$, which is unknown, and we cannot evaluate $L(z,h)$ for all $z$. $R_{emp}$ replaces the integral by a finite average - a **surrogate**.
> - We would like $h^*=\arg\min_{h\in H}R(h)$; training actually computes $\arg\min_{h\in H}R_{emp}(h,D_{TR})$. That substitution is **ERM**.
> - It is a *principle*, not a theorem: nothing guarantees the two argmins coincide. Whether $R_{emp}$ says anything about $R$ depends on the amount of data and on the capacity of $H$ - which is exactly what [[Statistical Learning Theory]] quantifies.
> - Corollary to never forget: there can exist $h'$ with $E'>E$ on TR but $R'<R$ - fitting better is not being better.
>
> See [[Generalization and Risk]].

### A3 [theory] Loss, Error, Risk

Distinguish the **inner loss**, the **Error/Risk** and the **Loss** minimized during training. Give the loss function used for regression, classification, clustering and density estimation.

> [!success]- Answer
> - **Inner loss** $L(h(\boldsymbol{x}),d)$: one pattern. **Error / Risk**: its average over a set, $E(\boldsymbol{w})=\frac1r\sum_p L(h_{\boldsymbol{w}}(\boldsymbol{x}_p),d_p)$ - always say *on which set*: $D_{TR}$, $D_{VL}$ or $D_{TS}$.
> - **Loss** (the training objective) $=$ data error $+$ penalty, e.g. Tikhonov. The number you report in tables and plots is the **Error**, i.e. the data term alone.
> - Regression: $L=(d_p-h(\boldsymbol{x}_p))^2$ $\to$ MSE. Classification: 0/1 loss, whose mean is the misclassification rate (piecewise constant $\to$ replaced at training time by a smooth surrogate, MSE or cross-entropy). Clustering / VQ: $L=\lVert\boldsymbol{x}_p-h(\boldsymbol{x}_p)\rVert^2$, squared distance from the centroid. Density estimation: $L=-\ln h(\boldsymbol{x}_p)$ (log-likelihood).
>
> See [[Loss Functions]], [[Notation]].

### A4 [exercise] Counting hypotheses in concept learning

Binary inputs, binary output, $n=5$.
(a) How many distinct Boolean functions are there?
(b) How many *semantically distinct* conjunctive hypotheses (literal, negated literal, don't-care)?
(c) Comment on the ratio.

> [!success]- Answer
> (a) $|H|=2^{2^n}=2^{32}\approx4.3\cdot10^{9}$.
> (b) $3^n+1=243+1=244$. Three choices per variable, plus $1$ for the single *false* hypothesis (any $h$ containing $l_i\wedge\lnot l_i$ collapses to it).
> (c) A reduction of seven orders of magnitude. That is the **language bias**: the [[Version Space]] becomes searchable without enumerating $H$, but any target outside the conjunctive language - e.g. `if (x1=1) or (x2=1)` - is now unreachable. Bias buys generalization and pays with representational loss.
>
> See [[Concept Learning]].

### A5 [derivation] An unbiased learner cannot generalize

Take $H=P(X)$, the power set of the instance space. Prove that the version space gives no answer on any unobserved instance.

> [!success]- Answer
> Let $D$ be the training data and $h\in VS(H,D)$, i.e. $h(\boldsymbol{x})=d(\boldsymbol{x})$ for every $(\boldsymbol{x},d)\in D$. Take any unobserved instance $\boldsymbol{x}_i$ and define
> $$h'(\boldsymbol{x})=h(\boldsymbol{x})\ \ \forall\boldsymbol{x}\neq\boldsymbol{x}_i,\qquad h'(\boldsymbol{x}_i)\neq h(\boldsymbol{x}_i).$$
> Since $H$ is the power set, $h'\in H$. Since $\boldsymbol{x}_i\notin D$, $h'$ agrees with $h$ on all the training data, so $h\in VS\Rightarrow h'\in VS$. Hypotheses in the VS therefore pair off: exactly half classify $\boldsymbol{x}_i$ positive and half negative, and majority voting over the VS is a tie. The learner can answer only "don't know" - it is a **lookup table**.
> Minimal instance: $X=\{0,1\}$, $H=\{x,\lnot x,0,1\}$, $D=\{(0,0)\}$ gives $VS=\{x,\ 0\}$, which disagrees on the input $1$.
> Conclusion: bias is needed for **generalization**, not merely for efficiency. Which bias is best is a different question, answered quantitatively by [[Statistical Learning Theory]].
>
> See [[Concept Learning]], [[Version Space]].

---

## Part B - Generalization

### B1 [derivation] Bias-variance decomposition

State the setting and derive
$$\mathbb{E}_P\big[(y-h(\boldsymbol{x}))^2\big]=\operatorname{Var}[h(\boldsymbol{x})]+\operatorname{Bias}[h(\boldsymbol{x})]^2+\sigma^2 .$$
Prove first the lemma you need.

> [!success]- Answer
> **Setting.** $y=f(\boldsymbol{x})+\varepsilon$ with $\mathbb{E}[\varepsilon]=0$, $\operatorname{Var}[\varepsilon]=\sigma^2$; data i.i.d. from $P$; the expectation $\mathbb{E}_P$ is over **all training sets** that could be drawn (and over the noise on the new point). Write $\bar h(\boldsymbol{x})=\mathbb{E}_P[h(\boldsymbol{x})]$.
>
> **Lemma.** $\operatorname{Var}[Z]=\mathbb{E}[(Z-\bar Z)^2]=\sum_i(z_i^2-2z_i\bar Z+\bar Z^2)P(z_i)=\mathbb{E}[Z^2]-2\bar Z^2+\bar Z^2=\mathbb{E}[Z^2]-\bar Z^2$, used as $\mathbb{E}[Z^2]=\bar Z^2+\operatorname{Var}[Z]$.
>
> **Step 1** - expand: $\mathbb{E}_P[(y-h)^2]=\mathbb{E}_P[h^2]+\mathbb{E}_P[y^2]-2\mathbb{E}_P[y]\mathbb{E}_P[h]$; the cross term factorizes because the noise on the new point is independent of the training set that produced $h$.
> **Step 2** - lemma on $h$: $\mathbb{E}_P[h^2]=\mathbb{E}_P[(h-\bar h)^2]+\bar h^2$.
> **Step 3** - lemma on $y$: zero-mean noise gives $\mathbb{E}_P[y]=f(\boldsymbol{x})$, so $\mathbb{E}_P[y^2]=\mathbb{E}_P[(y-f)^2]+f^2$.
> **Step 4** - substitute; the terms $\bar h^2-2f\bar h+f^2$ are the square $(\bar h-f)^2$:
> $$\mathbb{E}_P[(y-h)^2]=\underbrace{\mathbb{E}_P[(h-\bar h)^2]}_{\text{variance}}+\underbrace{(\bar h-f)^2}_{\text{bias}^2}+\underbrace{\mathbb{E}_P[(y-f)^2]}_{=\mathbb{E}[\varepsilon^2]=\sigma^2}$$
>
> **Reading.** Bias = systematic error of the *average* model, high when $H$ is too rigid. Variance = sensitivity to which training set you happened to get. Noise = irreducible, model-independent. Both bias and variance are computed **across training sets**, never from one run.
>
> See [[Bias-Variance Tradeoff]].

### B2 [exercise] Bias and variance by hand

The same model class, trained on three different training sets, predicts at a point $\boldsymbol{x}$ the values $h_1=2$, $h_2=4$, $h_3=6$. The true value is $f(\boldsymbol{x})=5$ and the noise variance is $\sigma^2=1$. Compute bias$^2$, variance and the expected squared error at $\boldsymbol{x}$.

> [!success]- Answer
> $\bar h=\frac{2+4+6}{3}=4$.
> $\operatorname{Bias}^2=(\bar h-f)^2=(4-5)^2=1$.
> $\operatorname{Var}=\frac{(2-4)^2+(4-4)^2+(6-4)^2}{3}=\frac{4+0+4}{3}=\frac83\approx2.67$.
> Expected error $=1+2.67+1=4.67$.
> Comment: this class is mostly hurt by **variance**, so the cure is more regularization / a smaller $H$ / [[Bagging]] - not a more flexible model.

### B3 [theory] The role of $\lambda$

Explain how the regularization coefficient $\lambda$ moves a model along the bias-variance trade-off, and why the "low $\lambda$" picture is the argument behind bagging. What warning applies when reading a plot indexed by $\lambda$?

> [!success]- Answer
> | $\lambda$ | effect |
> |---|---|
> | high | penalty dominates, small weights, rigid model: **high bias, low variance** $\to$ [[Underfitting]] |
> | medium | the trade-off point |
> | low | fits each particular training set: **low bias, high variance** $\to$ [[Overfitting]] |
>
> With very low $\lambda$ the individual fits are wild but their **mean** $\bar h(\boldsymbol{x})$ is still close to $f$. So averaging models trained on resampled data attacks the variance term directly and leaves the (already small) bias alone - that is exactly [[Bagging]].
> **Warning:** on an axis indexed by $\lambda$, complexity grows to the **left**, the reverse of the usual complexity plots. Read the axis before reading the curves.
>
> See [[Bias-Variance Tradeoff]], [[Regularization]].

### B4 [theory] Reading the error-vs-complexity plot

Describe the shape of training and test error curves as complexity grows, for many training sets of the same size. Where does the variance appear in the picture? What modern phenomenon breaks the classical reading?

> [!success]- Answer
> - Training curves fall monotonically and stay **bunched together**.
> - Test curves are **U-shaped** and **fan out** as complexity grows: the spread between a lucky and an unlucky training set *is* the variance.
> - The minimum of the average test curve is the trade-off point.
> - Caveat: the U-shape is about **capacity**, not parameter count. Pushed far past the interpolation threshold, test error can fall again - **double descent**, and its epoch-wise cousin **grokking** ([[Deep Learning]]). The plain VC reading does not predict this.
> - Practical consequence: a single train/test split can be lucky or unlucky $\to$ repeated splits, [[Cross Validation]], report mean $\pm$ std.

### B5 [theory] Bayes optimal classifier

Define it, define the Bayes rate, and explain in what sense [[k-NN]] approximates it.

> [!success]- Answer
> If $P(\boldsymbol{x},y)$ were known, the best possible rule is $h(\boldsymbol{x})=\arg\max_v P(v\mid\boldsymbol{x})$. Its error rate is the **Bayes rate**: the minimum achievable given the data distribution - the floor set by the noise term of the decomposition.
> It is a *reference point*, not an algorithm: the generating density is unknown.
> k-NN approximates it by two relaxations: (i) the conditional probability *at a point* is replaced by the conditional probability *in a neighbourhood* of the point; (ii) the probabilities are estimated by training-sample **proportions** inside that neighbourhood.
>
> See [[Bayes Optimal Classifier]].

### B6 [theory] No Free Lunch

State the theorem informally. If no method is universally best, what *does* distinguish good models from bad ones?

> [!success]- Answer
> There is no universally best learning method: if an algorithm beats others on some problems, it must pay with worse results on others. Averaged over all possible target functions, all learners are equivalent - because each learner is just a different **bias**, and a bias helps exactly on the problems that match it.
> What still separates models in practice:
> - **Flexibility** - can it approximate arbitrary functions in principle? (linear vs [[Neural Networks]])
> - **Control of complexity** - can that flexibility be reined in? ([[Regularization]], [[Structural Risk Minimization]])
>
> "Flexible $H$ + principled complexity control" is the core of ML. NFL also warns against claims like "deep always wins".
>
> See [[No Free Lunch]].

---

## Part C - Capacity and bounds

### C1 [theory] Shattering and VC dimension

Define dichotomy, shattering and $VC(H)$. What exactly does $VC(H)=p$ assert? Give the VC dimension of linear separators in $\mathbb{R}^n$, of intervals on $\mathbb{R}$, and of k-NN.

> [!success]- Answer
> With $N$ instances there are $2^N$ **dichotomies** ($\pm1$ labelings). A dichotomy is *represented* if some $h\in H$ realizes it with 0 errors. $H$ **shatters** a set iff it represents all its dichotomies.
> $VC(H)$ = maximum cardinality of a configuration of points that $H$ shatters. $VC(H)=p$ means:
> 1. $H$ shatters **at least one** configuration of $p$ points, **and**
> 2. $H$ shatters **no** configuration of $p+1$ points.
>
> (Note the asymmetry: "at least one" for $p$, "none" for $p+1$.) If arbitrarily large sets can be shattered, $VC=\infty$.
> Values: LTU in $\mathbb{R}^n$: $n+1$ (an upper bound - constraints can lower it). Intervals on $\mathbb{R}$: 2. Circles $\operatorname{sign}(\boldsymbol{x}\cdot\boldsymbol{x}-b)$ centred at the origin: 1; $\operatorname{sign}(k\boldsymbol{x}\cdot\boldsymbol{x}-b)$: 2; free circles in 2D: 3. [[k-NN]]: **infinite**.
>
> See [[VC Dimension]].

### C2 [derivation] $VC(\text{lines in }\mathbb{R}^2)=3$

Prove both halves.

> [!success]- Answer
> **$VC\ge3$.** Exhibit one configuration of 3 points in general position (a triangle). All $2^3=8$ labelings are realizable by a line. Note it is *not* required that every configuration works - 3 collinear points with the pattern $+,-,+$ cannot be separated, and that is irrelevant.
> **$VC<4$.** Take any 4 points and split into the two possible configurations.
> - *Convex position* (a quadrilateral): its two **diagonals cross**. Label the endpoints of one diagonal $+$ and the endpoints of the other $-$. The two classes' convex hulls share the intersection point, so no line can separate them - this is exactly XOR.
> - *One point inside the triangle* of the other three: label the inner point $+$ and the three outer ones $-$. The inner point lies in the convex hull of the negatives, so again the hulls intersect.
>
> In both cases a dichotomy is missing, so **no** configuration of 4 points is shattered. Recall the definition is asymmetric: for $p+1$ we must rule out *every* configuration, and we just did.
> Therefore $VC=3=n+1$ with $n=2$. $\blacksquare$

### C3 [exercise] Computing the VC-confidence

For the 0/1 loss the bound is, with probability $\ge1-\delta$ and $VC<N$,
$$R\le R_{emp}+\epsilon,\qquad \epsilon=\sqrt{\frac{VC\left(\ln\frac{2N}{VC}+1\right)-\ln\frac{\delta}{4}}{N}} .$$
With $N=1000$ and $\delta=0.05$, compute $\epsilon$ for (a) $VC=5$ and (b) $VC=50$. Model (a) has $R_{emp}=0.10$, model (b) has $R_{emp}=0$. Which does SRM pick?

> [!success]- Answer
> (a) $\ln(2000/5)=\ln400=5.9915$; $+1=6.9915$; $\times5=34.96$. $-\ln(0.05/4)=-\ln 0.0125=4.382$. Sum $=39.34$; $/1000=0.03934$; $\epsilon=\sqrt{0.03934}\approx\mathbf{0.198}$.
> Guaranteed risk $=0.10+0.198=\mathbf{0.298}$.
> (b) $\ln(2000/50)=\ln40=3.6889$; $+1=4.6889$; $\times50=234.4$; $+4.382=238.8$; $/1000=0.2388$; $\epsilon\approx\mathbf{0.489}$.
> Guaranteed risk $=0+0.489=\mathbf{0.489}$.
> **SRM picks (a)**, despite its worse training error - the extra capacity of (b) is not paid for by $N=1000$ examples.
> Two remarks: $\epsilon$ can be computed **before** training (it needs only $VC$, $N$, $\delta$); and $0.489$ on a binary task is worse than a coin, which shows how **loose** these worst-case bounds are - they are for ranking, not for assessment.
>
> See [[Statistical Learning Theory]].

### C4 [theory] Structural Risk Minimization

Describe SRM: the structure it needs, the two columns you tabulate, and the two practical ways of applying it. Which of the two is SVM?

> [!success]- Answer
> Define a **nested** family ordered by capacity, $H_1\subseteq H_2\subseteq\dots\subseteq H_n$ with $VC(H_1)\le\dots\le VC(H_n)$ - both conditions are needed: nesting is what makes "one step more complex" well defined.
> For each $H_i$ tabulate: $R_{emp}$ (measured after training, **decreasing** down the structure) and the VC-confidence (computed before training, **increasing**). Their sum has a minimum - pick that row, not the row with the lowest training error. The inductive principle: *the optimal complexity is the minimum of the bound*.
> Example structures: NN with growing number of hidden units (and epochs, and weight size - a NN is a **variable-size $H$**); polynomials of degree $M$; $\lVert\boldsymbol{w}\rVert<c$ with growing $c$, i.e. decreasing $\lambda$; decision trees with more nodes. The important hyperparameters are exactly those related to VC.
> Two practical approaches:
> 1. **Fix the complexity, minimize the training error** - what we do with NN (possibly with an implicit SRM through [[Early Stopping]], or the explicit Tikhonov loss).
> 2. **Fix the training error, minimize the capacity** - this is [[SVM]]: hard margin makes $R_{emp}=0$ by construction, and maximizing the margin minimizes the VC bound.
>
> See [[Structural Risk Minimization]].

### C5 [theory] VC dimension vs number of parameters; limits of the bound

Are they the same thing? For what is the VC bound good, and for what is it useless?

> [!success]- Answer
> **Not the same.** $\operatorname{sign}(\boldsymbol{x}\cdot\boldsymbol{x}-b)$ and $\operatorname{sign}(k\boldsymbol{x}\cdot\boldsymbol{x}-b)$ have different parameter counts and VC $1$ vs $2$; redundant parameters can always be added; there exist one-parameter families with infinite VC. For NN, capacity depends on the number of units **and on the size of the weights** - which is why weight decay lowers the effective VC.
> Related fact: for many reasonable classes (e.g. linear approximators) VC is linear in the number of free parameters, which is the origin of the "$k$ examples per parameter" rules of thumb.
> **Good for:** a model- and algorithm-independent theoretical ground; making complexity control an inductive principle instead of trial and error; **relative** comparison/ranking of models; driving the design of new models (SVM came from here).
> **Bad for:** actually estimating predictive error - the bounds are worst case and can be hundreds of times larger than the real overfitting effect, so they are unusable for model assessment; classes whose VC is hard to compute (non-linear parameters); over-parametrized regimes (double descent).
> Alternatives with the same *fit + complexity* shape: AIC $-2\log\hat L+2k$, BIC $-2\log\hat L+k\ln N$ (both need a likelihood and are limited to models linear in the parameters; BIC penalizes harder as $N$ grows), MDL (description length), Bayesian priors, Tikhonov ($\lVert\boldsymbol{w}\rVert$). They differ in *what they call complexity*, so they can disagree.

### C6 [theory] Cover's theorem

State it, list the models it justifies, and state the caveat.

> [!success]- Answer
> *A complex pattern-classification problem cast nonlinearly into a high-dimensional space is more likely to be linearly separable than in a low-dimensional space, provided the space is not densely populated.*
> Extreme deterministic case: lift $l$ examples onto the vertices of the $(l-1)$-dimensional simplex and **every** binary partition becomes linearly separable.
> It is the justification of the whole "expand, then separate linearly" family: [[Linear Basis Expansion]] (fixed $\phi$), [[Neural Networks]] (adaptive $\phi$), [[Randomized Neural Networks]] and [[Echo State Networks]] (random $\phi$), [[SVM]]/[[Kernel Methods]] (implicit $\phi$ via the kernel).
> **Caveat** - the last clause. Expanding without enough data is the [[Curse of Dimensionality]], and separability bought this way still has to be paid for by complexity control ([[VC Dimension]], [[Regularization]]).

---

## Part D - Linear models, SVM and kernels

### D1 [derivation] The normal equation

Starting from $E(\boldsymbol{w})=\lVert\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}\rVert^2$, derive the normal equation. Discuss the singular case, the tool used in practice, and the cost.

> [!success]- Answer
> $E(\boldsymbol{w})=\sum_p(y_p-\boldsymbol{x}_p^T\boldsymbol{w})^2$ is quadratic and convex, so a minimum exists (not necessarily unique) and there are no local minima. Setting the partial derivatives to zero:
> $$\frac{\partial E}{\partial w_j}=-2\sum_{p=1}^{l}(y_p-\boldsymbol{x}_p^T\boldsymbol{w})x_{p,j}=0\quad\forall j\ \Longrightarrow\ \boldsymbol{X}^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w})=\boldsymbol{0}\ \Longrightarrow\ (\boldsymbol{X}^T\boldsymbol{X})\boldsymbol{w}=\boldsymbol{X}^T\boldsymbol{y}$$
> - $\boldsymbol{X}^T\boldsymbol{X}$ non-singular: $\boldsymbol{w}=(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}=\boldsymbol{X}^{+}\boldsymbol{y}$ (Moore-Penrose pseudoinverse).
> - Singular (e.g. collinear features, $n>l$): infinitely many solutions - take the **minimum-norm** one.
> - Practical tool: **SVD**, $\boldsymbol{X}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T\Rightarrow\boldsymbol{X}^{+}=\boldsymbol{V}\boldsymbol{\Sigma}^{+}\boldsymbol{U}^T$ (reciprocate the nonzero singular values); $\boldsymbol{w}=\boldsymbol{X}^{+}\boldsymbol{y}$ is directly the minimal-norm least-squares solution.
> - Cost cubic in the matrix dimension $\to$ [[Gradient Descent]] is usually preferred, and it is also compatible with regularization, with early stopping and with non-linear models.
>
> See [[Normal Equation]], [[Linear Models]].

### D2 [exercise] Least squares and ridge by hand

Data: $(1,1),(2,2),(3,2)$; model $h(x)=w_0+w_1x$.
(a) Find $w_0,w_1$ by least squares. (b) Compute the residuals and the MSE. (c) Redo the slope with an L2 penalty $\lambda=1$ on the slope only (centred data). Comment.

> [!success]- Answer
> (a) $\bar x=2$, $\bar y=5/3$. $S_{xy}=\sum(x_p-\bar x)(y_p-\bar y)=(-1)(-\tfrac23)+0+(1)(\tfrac13)=1$; $S_{xx}=1+0+1=2$.
> $w_1=S_{xy}/S_{xx}=\mathbf{0.5}$, $w_0=\bar y-w_1\bar x=\tfrac53-1=\mathbf{2/3}\approx0.667$.
> (b) Predictions $7/6,\ 5/3,\ 13/6$; residuals $-1/6,\ +1/3,\ -1/6$. $SSE=\tfrac1{36}+\tfrac4{36}+\tfrac1{36}=\tfrac16$; $MSE=\tfrac1{18}\approx0.056$.
> (c) $w_1^{ridge}=\dfrac{S_{xy}}{S_{xx}+\lambda}=\dfrac{1}{3}\approx0.333$.
> The slope **shrinks toward zero**: the penalty trades a little data error for a smaller weight norm, i.e. lower capacity. The intercept is deliberately left unpenalized, otherwise the solution would depend on a shift of the targets.
>
> See [[Ridge Regression]].

### D3 [derivation] Ridge regression

Derive $\boldsymbol{w}=(\boldsymbol{X}^T\boldsymbol{X}+\lambda\boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y}$, argue that the matrix is always invertible, and show that the gradient version is **weight decay**.

> [!success]- Answer
> $\operatorname{Loss}(\boldsymbol{w})=\lVert\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}\rVert^2+\lambda\lVert\boldsymbol{w}\rVert^2$. Gradient: $-2\boldsymbol{X}^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w})+2\lambda\boldsymbol{w}=\boldsymbol{0}\Rightarrow(\boldsymbol{X}^T\boldsymbol{X}+\lambda\boldsymbol{I})\boldsymbol{w}=\boldsymbol{X}^T\boldsymbol{y}$.
> **Invertibility.** $\boldsymbol{X}^T\boldsymbol{X}$ is symmetric positive semi-definite, so its eigenvalues are $\mu_i\ge0$; those of $\boldsymbol{X}^T\boldsymbol{X}+\lambda\boldsymbol{I}$ are $\mu_i+\lambda>0$ for $\lambda>0$. All eigenvalues strictly positive $\Rightarrow$ the matrix is positive definite, hence non-singular. Ridge therefore also *fixes* the degenerate least-squares case.
> **Gradient version.** Differentiating the two terms separately and using $\eta$ only on the error term:
> $$\boldsymbol{w}_{new}=\boldsymbol{w}+\eta\,\Delta\boldsymbol{w}-2\lambda\boldsymbol{w}$$
> i.e. every step shrinks the current weights by a constant factor before adding the gradient - **weight decay**. Keeping $\eta$ off $\lambda$ keeps the two hyperparameters independent and easier to select.
> $\lambda$ is a hyperparameter, always chosen on validation data. High $\lambda$: rigid model, [[Underfitting]]; low $\lambda$: [[Overfitting]].

### D4 [theory] LMS vs Perceptron

Both train a linear separator. Why does LMS *not* minimize the number of misclassifications, and what does the perceptron do differently?

> [!success]- Answer
> The LTU output is $h(\boldsymbol{x})=\operatorname{sign}(\boldsymbol{w}^T\boldsymbol{x})$, but its 0/1 error is piecewise constant, hence not differentiable. LMS therefore takes the gradient of the squared error computed **before** the threshold, on $\boldsymbol{w}^T\boldsymbol{x}$: its delta is $(d-\boldsymbol{w}^T\boldsymbol{x})$, which is generally nonzero even for **correctly classified** patterns. So LMS keeps moving the hyperplane to reduce a regression error and can end up misclassifying points even on a linearly separable problem.
> The perceptron's delta is $(d-\operatorname{sign}(\boldsymbol{w}^T\boldsymbol{x}))$: zero on correct patterns, so only mistakes move the weights, $\boldsymbol{w}_{new}=\boldsymbol{w}+\eta\,d\,\boldsymbol{x}$. It targets misclassifications directly and converges in finite time when the data are separable - but it may cycle when they are not, whereas LMS always converges to *something*.
> Note also that many separating hyperplanes exist; neither algorithm picks a canonical one. That is what [[SVM]] adds.

### D5 [derivation] Perceptron convergence theorem

State and prove it. Then interpret the final bound $q\le(R/\gamma)^2$.

> [!success]- Answer
> **Statement.** For a linearly separable training set, the perceptron algorithm makes only a finite number of weight updates.
> **Assumptions.** $\boldsymbol{w}(0)=\boldsymbol{0}$, $\eta=1$, and $\exists\boldsymbol{w}^*$ with $\boldsymbol{w}^{*T}\boldsymbol{x}_i>0$ for all $i$. WLOG all targets are $+1$: replacing $\boldsymbol{x}_i$ by $\boldsymbol{x}_i'=d_i\boldsymbol{x}_i$ shows $\boldsymbol{w}^*$ solves one problem iff it solves the other, so every mistake is a false negative. $q$ counts **mistakes** (= updates), not patterns or epochs. Constants: $\alpha=\min_i\boldsymbol{w}^{*T}\boldsymbol{x}_i>0$ (smallest projection on the solution) and $\beta=\max_i\lVert\boldsymbol{x}_i\rVert^2$ (data radius squared).
> After $q$ mistakes, $\boldsymbol{w}(q)=\sum_{j=1}^{q}\boldsymbol{x}_i(j)$.
>
> **Lower bound (alignment).** $\boldsymbol{w}^{*T}\boldsymbol{w}(q)=\sum_j\boldsymbol{w}^{*T}\boldsymbol{x}_i(j)\ge q\alpha$. By **Cauchy-Schwarz**, $\lVert\boldsymbol{w}^*\rVert^2\lVert\boldsymbol{w}(q)\rVert^2\ge(\boldsymbol{w}^{*T}\boldsymbol{w}(q))^2\ge q^2\alpha^2$, hence
> $$\lVert\boldsymbol{w}(q)\rVert^2\ \ge\ \frac{q^2\alpha^2}{\lVert\boldsymbol{w}^*\rVert^2}.$$
> **Upper bound (length).** $\lVert\boldsymbol{w}(q)\rVert^2=\lVert\boldsymbol{w}(q-1)\rVert^2+2\boldsymbol{w}(q-1)^T\boldsymbol{x}_i(q)+\lVert\boldsymbol{x}_i(q)\rVert^2$. The cross term is $\le0$ **precisely because $\boldsymbol{x}_i(q)$ was misclassified** ($\boldsymbol{w}(q-1)^T\boldsymbol{x}_i(q)\le0$). Dropping it and iterating from $\boldsymbol{w}(0)=\boldsymbol{0}$:
> $$\lVert\boldsymbol{w}(q)\rVert^2\ \le\ \sum_{j=1}^{q}\lVert\boldsymbol{x}_i(j)\rVert^2\ \le\ q\beta.$$
> **Squeeze.** $q\beta\ge q^2\alpha^2/\lVert\boldsymbol{w}^*\rVert^2\Rightarrow q\le\beta\lVert\boldsymbol{w}^*\rVert^2/\alpha^2$. A parabola cannot stay under a line forever, so $q$ is finite. $\blacksquare$
>
> **Interpretation.** With $\gamma=\alpha/\lVert\boldsymbol{w}^*\rVert$ (the *geometric* distance of the closest pattern to the hyperplane, which unlike $\alpha$ is scale-invariant) and $R=\sqrt\beta$:
> $$q\le\left(\frac{R}{\gamma}\right)^2=\left(\frac{\text{size of the data}}{\text{width of the gap}}\right)^2 .$$
> Neither the number of patterns nor the input dimension appears - only geometry. The bound is tightest for the $\boldsymbol{w}^*$ with the largest $\gamma$, i.e. the **maximum-margin** separator - the idea SVM is built on. And $\gamma$ is squared in the denominator, so "finite" can still mean impractically many steps.

### D6 [exercise] Perceptron by hand

$\boldsymbol{w}(0)=(0,0)$, $\eta=1$, patterns $\boldsymbol{x}_1=(1,1)$ with $d_1=+1$ and $\boldsymbol{x}_2=(1,-1)$ with $d_2=-1$ (bias already absorbed). Convention: a mistake occurs when $d_i\,\boldsymbol{w}^T\boldsymbol{x}_i\le0$. Run the algorithm until convergence and count the updates.

> [!success]- Answer
> **Step 1** - $\boldsymbol{w}^T\boldsymbol{x}_1=0\le0$ with $d=+1$: mistake. $\boldsymbol{w}\leftarrow(0,0)+(1,1)=(1,1)$.
> **Step 2** - $\boldsymbol{w}^T\boldsymbol{x}_2=1-1=0$, $d_2\cdot0=0\le0$: mistake. $\boldsymbol{w}\leftarrow(1,1)-(1,-1)=(0,2)$.
> **Check** - $\boldsymbol{x}_1$: $(0,2)\cdot(1,1)=2>0$, $d=+1$ ✓. $\boldsymbol{x}_2$: $(0,2)\cdot(1,-1)=-2<0$, $d=-1$ ✓.
> Converged after **$q=2$ updates**; the boundary is $x_2=0$.
> Sanity check against the bound: $\boldsymbol{w}^*=(0,1)$ works, $\alpha=\min(1,1)=1$, $\lVert\boldsymbol{w}^*\rVert^2=1$, $\beta=2$, so $q\le2$ - the run is exactly at the bound.

### D7 [derivation] The margin is $2/\lVert\boldsymbol{w}_o\rVert$

Prove it, and explain why we minimize $\tfrac12\lVert\boldsymbol{w}\rVert^2$ instead of maximizing $\rho$.

> [!success]- Answer
> $\boldsymbol{w}_o$ is orthogonal to the hyperplane $g(\boldsymbol{x})=\boldsymbol{w}_o^T\boldsymbol{x}+b_o=0$. Decompose any point as $\boldsymbol{x}=\boldsymbol{x}_p+r\frac{\boldsymbol{w}_o}{\lVert\boldsymbol{w}_o\rVert}$ with $\boldsymbol{x}_p$ on the hyperplane and $r$ the signed distance. Then
> $$g(\boldsymbol{x})=\underbrace{g(\boldsymbol{x}_p)}_{=0}+r\,\boldsymbol{w}_o^T\frac{\boldsymbol{w}_o}{\lVert\boldsymbol{w}_o\rVert}=r\lVert\boldsymbol{w}_o\rVert\quad\Longrightarrow\quad r=\frac{g(\boldsymbol{x})}{\lVert\boldsymbol{w}_o\rVert}.$$
> In the **canonical representation** the closest points satisfy $|g(\boldsymbol{x}^{(s)})|=1$ - a legitimate normalization, since scaling $(\boldsymbol{w},b)$ by $K>0$ leaves the boundary unchanged. So a support vector sits at distance $1/\lVert\boldsymbol{w}_o\rVert$, and the margin (both sides) is
> $$\rho=\frac{2}{\lVert\boldsymbol{w}_o\rVert}.$$
> Maximizing $\rho$ $\iff$ minimizing $\lVert\boldsymbol{w}\rVert$ $\iff$ minimizing $\tfrac12\lVert\boldsymbol{w}\rVert^2$: same argmin, but the squared form is differentiable everywhere and quadratic (so the problem is a convex QP), and the $\tfrac12$ cancels the $2$ from the derivative.

### D8 [derivation] Primal, dual and sparsity

Write the hard-margin primal, build the Lagrangian, derive the dual, and use the Kuhn-Tucker conditions to prove that the solution is sparse.

> [!success]- Answer
> **Primal.** $\min\ \Psi(\boldsymbol{w})=\tfrac12\boldsymbol{w}^T\boldsymbol{w}$ s.t. $d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\ge1\ \forall i$.
> **Lagrangian**, one multiplier $\alpha_i\ge0$ per constraint:
> $$J(\boldsymbol{w},b,\boldsymbol{\alpha})=\tfrac12\boldsymbol{w}^T\boldsymbol{w}-\sum_i\alpha_i\big(d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)-1\big)$$
> to be minimized in $(\boldsymbol{w},b)$ and maximized in $\boldsymbol{\alpha}$ - the solution is a **saddle point**.
> **Stationarity.** $\partial J/\partial\boldsymbol{w}=0\Rightarrow\boldsymbol{w}=\sum_i\alpha_id_i\boldsymbol{x}_i$ (the *representer* result: the optimum is a linear combination of the training points); $\partial J/\partial b=0\Rightarrow\sum_i\alpha_id_i=0$.
> **Dual.** Substituting both back eliminates $\boldsymbol{w}$ and $b$:
> $$\max_{\boldsymbol{\alpha}}\ Q(\boldsymbol{\alpha})=\sum_i\alpha_i-\frac12\sum_i\sum_j\alpha_i\alpha_jd_id_j\,\boldsymbol{x}_i^T\boldsymbol{x}_j\quad\text{s.t.}\quad\alpha_i\ge0,\ \sum_i\alpha_id_i=0 .$$
> Two consequences: the data enter **only through dot products** (the door to [[Kernel Methods]]), and the cost scales with the number of patterns $N$, not with the dimension $m$.
> **Sparsity (KT complementarity).** At the optimum $\alpha_i\big(d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)-1\big)=0$ for every $i$. A product vanishes only if a factor does:
> - $\alpha_i>0\Rightarrow d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)=1$: $\boldsymbol{x}_i$ **is** a support vector, on the margin edge;
> - $\boldsymbol{x}_i$ not on the edge $\Rightarrow$ its constraint is strict $\Rightarrow\alpha_i=0$.
>
> Hence $\boldsymbol{w}_o=\sum_{i=1}^{N_s}\alpha_{o,i}d_i\boldsymbol{x}_i$ depends **only on the support vectors**. $\blacksquare$

### D9 [derivation] Soft margin

Add slack variables. Show that the only trace they leave in the dual is a box constraint, and classify the training points by the value of $\alpha_i$. How is $b_o$ recovered, and which points may *not* be used for it?

> [!success]- Answer
> **Primal.** $\min\ \tfrac12\boldsymbol{w}^T\boldsymbol{w}+C\sum_i\xi_i$ s.t. $d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\ge1-\xi_i$, $\xi_i\ge0$.
> Two families of multipliers: $\alpha_i$ for the margin constraints, $\mu_i$ for $\xi_i\ge0$. Differentiating the Lagrangian in $\xi_i$:
> $$C-\alpha_i-\mu_i=0\ \Longrightarrow\ \alpha_i=C-\mu_i,\qquad \mu_i\ge0\ \Longrightarrow\ \boxed{0\le\alpha_i\le C}$$
> The $\xi_i$ cancel out; the dual objective is **identical** to the hard-margin one, only the box changes.
> **Position of the points** (from $\alpha_i(d_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)+\xi_i-1)=0$ and $\mu_i\xi_i=0$):
>
> | $\alpha_i$ | $\mu_i$ | $\xi_i$ | where |
> |---|---|---|---|
> | $0$ | $>0$ | $0$ | outside the margin, not a SV |
> | $0<\alpha_i<C$ | $>0$ | $0$ | **on the margin edge** |
> | $C$ | $0$ | $\ge0$ | inside the margin, or misclassified |
>
> **Recovering $b_o$.** It vanished from the dual. Take any $j$ with $0<\alpha_j<C$: such a point has $\xi_j=0$ and lies exactly on the edge, so $d_j(\boldsymbol{w}_o^T\boldsymbol{x}_j+b_o)=1$; multiplying by $d_j$ (with $d_j^2=1$),
> $$b_o=d_j-\sum_i\alpha_{o,i}d_i\,k(\boldsymbol{x}_i,\boldsymbol{x}_j),$$
> averaged over all valid $j$ for numerical stability. Points with $\alpha_j=C$ **must not** be used: they may have $\xi_j>0$, their constraint is not tight, and they give the wrong $b$.
> Role of $C$: low $C$ tolerates errors $\to$ possible [[Underfitting]]; high $C$ forbids them $\to$ smaller margin $\to$ possible [[Overfitting]]; $C\to\infty$ recovers the hard margin. Note that admitting points inside the margin also *buys a larger margin*.

### D10 [exercise] A two-point SVM

Hard margin, $\boldsymbol{x}_1=(1,1)$ with $d_1=+1$ and $\boldsymbol{x}_2=(-1,-1)$ with $d_2=-1$. Find $\boldsymbol{w}_o$, $b_o$, the margin $\rho$ and the multipliers $\alpha_i$. Verify strong duality.

> [!success]- Answer
> By symmetry $b_o=0$ and $\boldsymbol{w}_o=(a,a)$. The canonical constraint on $\boldsymbol{x}_1$ is $a+a\ge1$, tight at the optimum: $a=1/2$, so $\boldsymbol{w}_o=(0.5,0.5)$, $\lVert\boldsymbol{w}_o\rVert=\sqrt{0.5}\approx0.707$.
> $$\rho=\frac{2}{\lVert\boldsymbol{w}_o\rVert}=2\sqrt2\approx2.83$$
> which is exactly the distance between the two points - as it must be, since both are support vectors and the hyperplane bisects the segment.
> **Multipliers.** $\boldsymbol{w}_o=\alpha_1(+1)(1,1)+\alpha_2(-1)(-1,-1)=(\alpha_1+\alpha_2)(1,1)$, so $\alpha_1+\alpha_2=0.5$; the constraint $\sum_i\alpha_id_i=0$ gives $\alpha_1=\alpha_2$, hence $\alpha_1=\alpha_2=\mathbf{0.25}$.
> **Duality check.** $\boldsymbol{x}_1^T\boldsymbol{x}_1=\boldsymbol{x}_2^T\boldsymbol{x}_2=2$, $\boldsymbol{x}_1^T\boldsymbol{x}_2=-2$.
> $Q=\sum\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_jd_id_j\boldsymbol{x}_i^T\boldsymbol{x}_j=0.5-\tfrac12(0.125+0.125+2\cdot0.125)=0.5-0.25=0.25$,
> and the primal value is $\tfrac12\lVert\boldsymbol{w}_o\rVert^2=\tfrac12\cdot0.5=0.25$. They coincide ✓.

### D11 [derivation] The kernel trick

Define an inner-product kernel and state Mercer's condition. Verify explicitly that $k(\boldsymbol{x},\boldsymbol{y})=(\boldsymbol{x}^T\boldsymbol{y})^2$ corresponds to $\Phi(\boldsymbol{x})=(x_1^2,\sqrt2x_1x_2,x_2^2)^T$. Derive the distance induced by a kernel and list the closure properties.

> [!success]- Answer
> $k(\boldsymbol{x},\boldsymbol{y})=\Phi^T(\boldsymbol{x})\Phi(\boldsymbol{y})$, symmetric; collecting all pairs gives the $N\times N$ Gram matrix $\boldsymbol{K}$.
> **Mercer:** $k$ is an inner product in *some* feature space **iff** its kernel matrix is symmetric **positive semi-definite** on every finite sample (non-negative eigenvalues).
> **Verification.**
> $$\Phi^T(\boldsymbol{x})\Phi(\boldsymbol{y})=x_1^2y_1^2+2x_1x_2y_1y_2+x_2^2y_2^2=(x_1y_1+x_2y_2)^2=(\boldsymbol{x}^T\boldsymbol{y})^2$$
> A dot product in a 3-dimensional feature space computed with a 2-dimensional operation on the inputs. Nothing was mapped: the middle expression only shows the two ends agree.
> **Induced distance:** $d_k(\boldsymbol{s},\boldsymbol{t})=\lVert\Phi(\boldsymbol{s})-\Phi(\boldsymbol{t})\rVert^2=k(\boldsymbol{s},\boldsymbol{s})-2k(\boldsymbol{s},\boldsymbol{t})+k(\boldsymbol{t},\boldsymbol{t})$ - just the expansion of $\lVert\boldsymbol{a}-\boldsymbol{b}\rVert^2$. Similar objects $\to$ small distance $\to$ high kernel value: a kernel *is* a similarity measure.
> **Closure:** if $k_1,k_2$ are kernels, so are $k_1+k_2$, $\alpha k_1$ ($\alpha>0$) and $k_1k_2$ - the cheapest way to prove a new kernel valid without touching Mercer.
> Standard kernels: polynomial $(\boldsymbol{x}^T\boldsymbol{x}_i+1)^p$; RBF $e^{-\lVert\boldsymbol{x}-\boldsymbol{x}_i\rVert^2/2\sigma^2}$ (infinite-dimensional feature space); two-layer perceptron $\tanh(\beta_0\boldsymbol{x}^T\boldsymbol{x}_i+\beta_1)$ - a valid kernel only for some $\beta_0,\beta_1$.
> The point: the background knowledge (the kernel) is **separated from the solver**, and the inputs need not be vectors at all - kernels for strings, trees and graphs exist.

### D12 [theory] Four misconceptions about SVM

Answer briefly: (a) can SVM be used with default hyperparameters? (b) is SVM immune to overfitting? (c) does it solve the curse of dimensionality? (d) does it find the number of units automatically?

> [!success]- Answer
> (a) **No.** $C$ and the kernel parameters must be selected together on validation data - they interact. A narrow RBF (small $\sigma$) needs a lower $C$; searching them sequentially introduces bias.
> (b) **No.** The clean SRM argument holds only for the **hard margin without kernel**. With soft margin Vapnik's theorem no longer applies ($C$ reappears as a hyperparameter), and an RBF kernel with $\sigma\to0$ behaves like 1-NN and has **infinite VC dimension** unless regularized.
> (c) **No.** SVM copes well with a high-dimensional **feature** space (complexity is ruled by the margin, not by the dimension), but the [[Curse of Dimensionality]] lives in the **input** space and is untouched.
> (d) Only in the picture-view sense: in the architecture reading, the number of hidden "units" equals the number of support vectors - but that is what the QP happens to leave nonzero, not a capacity search. And those units are **fixed** basis functions, unlike the *adaptive* hidden units of a NN.
> Honest extra: the number of SVs is not always small (≈42-45% of the data in the classic mixture example), and SVM is a **batch** algorithm.

### D13 [theory] SVM regression

Write the $\epsilon$-insensitive loss, describe the tube, and say which points become support vectors.

> [!success]- Answer
> $$L_\epsilon(d,y)=\begin{cases}\lvert d-y\rvert-\epsilon & \text{if }\lvert d-y\rvert\ge\epsilon\\ 0 & \text{otherwise}\end{cases}$$
> A flat bottom of width $2\epsilon$, then linear growth - the *soft margin loss*. It draws a **tube** of half-width $\epsilon$ around the prediction: points inside cost nothing.
> Two slack variables per pattern, one per side: $-\xi_i'-\epsilon\le d_i-\boldsymbol{w}^T\Phi(\boldsymbol{x}_i)\le\epsilon+\xi_i$, with $\xi_i,\xi_i'\ge0$.
> Only the points **on or outside** the tube become support vectors and guide the solution - exactly the mirror of classification, where the ignored points are those beyond the margin. Hyperparameters: $C$, $\epsilon$, and the kernel parameters.

---

## Part E - Neural networks and training

### E1 [theory] Why non-linear hidden units, and what XOR really shows

Prove that an MLP with linear hidden units is useless, then explain what the two-layer XOR solution demonstrates.

> [!success]- Answer
> With linear hidden units, $h(\boldsymbol{x})=\boldsymbol{W}^{out}(\boldsymbol{W}\boldsymbol{x})=(\boldsymbol{W}^{out}\boldsymbol{W})\boldsymbol{x}=\tilde{\boldsymbol{W}}\boldsymbol{x}$: a composition of linear maps is linear, so the network collapses to a single-layer model with the same $H$. Everything depth buys depends on $f$ being non-linear.
> **XOR.** A single LTU cannot do it (not linearly separable). With two hidden units $h_1=\mathrm{AND}(x_1,x_2)$ ($w_1=w_2=1$, $w_0=-1.5$) and $h_2=\mathrm{OR}(x_1,x_2)$ ($w_0=-0.5$), the output computes $\lnot h_1\wedge h_2$, and by De Morgan $(\lnot a\vee\lnot b)\wedge(a\vee b)=(\lnot a\wedge b)\vee(a\wedge\lnot b)=a\oplus b$.
> The point is **not** the logical trick but the **internal representation**: $(1,0)$ becomes $(h_1,h_2)=(0,1)$, and in that new space the problem *is* linearly separable. The hidden layer re-represents the input so the output layer's job becomes easy - the seed of [[Representation Learning]] and [[Deep Learning]], and an instance of [[Cover's Theorem]].

### E2 [theory] Universal approximation

State the theorem and say precisely what it does *not* give. If one hidden layer suffices, why use more?

> [!success]- Answer
> A network with a **single hidden layer** of sigmoidal units and a linear output can approximate arbitrarily well any continuous function on a hypercube, given enough hidden units: $\forall\epsilon\ \exists h$ with $|f(\boldsymbol{x})-h(\boldsymbol{x})|<\epsilon$, where $h(\boldsymbol{x})=f_k(\sum_jw_{kj}f_j(\sum_iw_{ji}x_i))$, $f_k$ linear. (Think of a finite Fourier series.)
> It is an **existence theorem**: it gives neither the learning algorithm ($\to$ [[Backpropagation]]) nor the number of units ($\to$ [[Model Selection and Assessment]], [[Cascade Correlation]]), and "enough units" may mean exponentially many.
> Depth is about **efficiency, not expressivity**: *no-flattening* results say there are functions approximable efficiently at depth $>d$ that require an exponentially larger model at depth $\le d$ - and depth may also help *learnability*. See E9.

### E3 [derivation] Backpropagation

Derive $\Delta_pw_{tu}=\delta_to_u$ with the delta for an output unit and for a hidden unit. Explain where the efficiency comes from.

> [!success]- Answer
> Forward: $net_t=\sum_sw_{ts}o_s$, $o_t=f_t(net_t)$; per-pattern error $E_p=\tfrac12\sum_{k=1}^{K}(d_k-o_k)^2$, $E_{tot}=\sum_pE_p$.
> Chain rule through $net_t$:
> $$\Delta_pw_{tu}=-\frac{\partial E_p}{\partial w_{tu}}=-\frac{\partial E_p}{\partial net_t}\frac{\partial net_t}{\partial w_{tu}}=\delta_t\,o_u,\qquad \frac{\partial net_t}{\partial w_{tu}}=o_u,\qquad \delta_t=-\frac{\partial E_p}{\partial o_t}f_t'(net_t)$$
> **Output unit** ($t=k$): $-\partial E_p/\partial o_k=(d_k-o_k)$, hence
> $$\delta_k=(d_k-o_k)\,f_k'(net_k).$$
> **Hidden unit** ($t=j$): $E_p$ depends on $o_j$ only through the units $k$ above it, so
> $$-\frac{\partial E_p}{\partial o_j}=\sum_k-\frac{\partial E_p}{\partial net_k}\frac{\partial net_k}{\partial o_j}=\sum_k\delta_kw_{kj}\ \Longrightarrow\ \delta_j=\Big(\sum_k\delta_kw_{kj}\Big)f_j'(net_j).$$
> Update: $w_{tu}^{new}=w_{tu}+\eta\,\delta_to_u$ (on-line; sum over $p$ for batch). With $m$ hidden layers the same derivation applies, the deltas coming from *any unit the current one feeds*.
> **Efficiency.** $O(\#W)$ instead of $O(\#W^2)$: the same $\delta_t$, computed once, is reused for all the weights entering unit $t$, and $\delta_k$ computed at the output is *reused* at the layer below - hence "back-propagation". With $10^9$ weights that is $10^9$ vs $10^{18}$ operations.
> Properties: **compositional** (follows the model's structure), **local** (each unit only needs quantities around it, yet the effect on the loss is global), and it requires a differentiable loss and differentiable activations. It is a **training algorithm**, not a type of network - never say "backprop NN". Unfold an RNN in time and the same algorithm is BPTT; the same holds for the shared kernels of a CNN.

### E4 [exercise] One backprop step

Network: 1 input $\to$ 1 hidden unit (logistic) $\to$ 1 linear output. $x=1$, $w_h=0.5$ (input$\to$hidden), $w_o=1.0$ (hidden$\to$output), no biases, target $d=1$, $\eta=0.5$. Do one on-line update.

> [!success]- Answer
> **Forward.** $net_h=0.5$; $o_h=\sigma(0.5)=1/(1+e^{-0.5})=0.6225$; $o=w_o\,o_h=0.6225$.
> **Output delta** (linear, $f'=1$): $\delta_o=(d-o)\cdot1=0.3775$. $\Delta w_o=\delta_o\,o_h=0.3775\times0.6225=0.2350$.
> **Hidden delta** (logistic, $f'=o_h(1-o_h)=0.6225\times0.3775=0.2350$):
> $\delta_h=(\delta_ow_o)f'(net_h)=0.3775\times1\times0.2350=0.0887$. $\Delta w_h=\delta_h\,x=0.0887$.
> **Update.** $w_o\leftarrow1+0.5(0.2350)=\mathbf{1.1175}$; $w_h\leftarrow0.5+0.5(0.0887)=\mathbf{0.5444}$.
> Check: both weights grew, as they should - the output was below target with a positive input.

### E5 [derivation] Activation derivatives and saturation

Show $f_\sigma'=f_\sigma(1-f_\sigma)$ and $\tanh'=1-\tanh^2$. Then explain why saturation stalls learning and what two practical rules follow.

> [!success]- Answer
> $f_\sigma(x)=(1+e^{-x})^{-1}$; $f_\sigma'(x)=e^{-x}(1+e^{-x})^{-2}=\dfrac{1}{1+e^{-x}}\cdot\dfrac{e^{-x}}{1+e^{-x}}=f_\sigma(1-f_\sigma)$.
> $\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$; by the quotient rule $\tanh'=\frac{(e^x+e^{-x})^2-(e^x-e^{-x})^2}{(e^x+e^{-x})^2}=1-\tanh^2$.
> **Saturation.** Every delta is gated by $f'$: $\delta=(d-o)f'(net)$. $f'$ is **maximum near $net=0$** (the near-linear zone) and **tiny in the saturated tails**, so a saturated unit receives almost no update, however wrong it is. In deep stacks these factors multiply $\to$ vanishing gradient.
> Two rules: **start with small weights**, so units begin in the linear zone ([[Weight Initialization]]); and **keep the targets inside the output range** (use $0.9$ / $-0.9$ rather than $1$ / $-1$), otherwise the unit chases an asymptote it can never reach. The threshold unit is excluded from gradient training altogether - its derivative does not exist.

### E6 [derivation + exercise] Softmax and cross-entropy

Show $\partial o_k/\partial net_j=o_k(\delta_{kj}-o_j)$. Then, for $\boldsymbol{net}=(1,2,0)$, compute the softmax outputs and the cross-entropy if the true class is the **first** one.

> [!success]- Answer
> $o_k=e^{net_k}/\sum_je^{net_j}$. For $j=k$: $\partial o_k/\partial net_k=\frac{e^{net_k}S-e^{net_k}e^{net_k}}{S^2}=o_k(1-o_k)$ with $S=\sum_je^{net_j}$. For $j\neq k$: $\partial o_k/\partial net_j=-\frac{e^{net_k}e^{net_j}}{S^2}=-o_ko_j$. Both cases: $o_k(\delta_{kj}-o_j)$.
> **Numbers.** $e^1=2.7183$, $e^2=7.3891$, $e^0=1$; $S=11.1074$.
> $\boldsymbol{o}=(0.2447,\ 0.6652,\ 0.0900)$ - positive and summing to 1, so readable as $p(\text{class}=k\mid\boldsymbol{x})$.
> Cross-entropy for true class 1: $-\ln(0.2447)=\mathbf{1.408}$.
> (Had the prediction been correct and confident, e.g. $o=0.99$, the loss would be $0.01$; the loss is $0$ only for a perfect one-hot output.) Softmax + cross-entropy is the standard output pairing for multiclass; the alternatives are 1-of-K with sigmoidal outputs, OVA and AVA - see [[Multiclass Classification]].

### E7 [theory] Batch, on-line and mini-batch

Compare them: gradient quality, speed, stability, interaction with $\eta$, and the reason mini-batch is what scales.

> [!success]- Answer
> An **epoch** is one full presentation of the training set.
> - **Batch:** sum the gradients over all $l$ patterns, then update. Accurate gradient estimate, parallelizable, but you wait for all the data.
> - **On-line / SGD:** update after each pattern; the single-pattern gradient is a noisy estimate $\to$ zig-zag descent. Progresses with every example, needs a **smaller $\eta$**, can help escape local minima, and its noise acts as a mild regularizer. Must **shuffle** every epoch, otherwise a fixed order biases the descent.
> - **Mini-batch:** $1<mb<l$; the compromise, with $mb$ chosen to fit GPU memory, plus random sampling for an unbiased estimate.
> **$\eta$ interaction:** dividing the gradient by $l$ (LMS) or by $mb$ rescales the effective $\eta$, so comparing batch and on-line *at the same $\eta$ is meaningless*. Same for $\lambda$: to compare fairly use $\lambda\times(mb/l)$ - or just let model selection handle it.
> **Scaling:** the cost of one SGD update depends on $mb$, not on $l$. That is what extends learning to millions of examples (and to streams, where there are no epochs at all).

### E8 [theory] Momentum, learning rate, initialization, minima

Write the momentum update and its Nesterov variant. Then explain how $\eta$, $\boldsymbol{w}_{initial}$, momentum and the stopping point together constitute a **control over the search in $H$**.

> [!success]- Answer
> $$\Delta\boldsymbol{w}_{new}=-\eta\frac{\partial E}{\partial\boldsymbol{w}}+\alpha\,\Delta\boldsymbol{w}_{old},\qquad\boldsymbol{w}_{new}=\boldsymbol{w}+\Delta\boldsymbol{w}_{new},\qquad 0<\alpha<1\ \ (\text{typ. }0.5-0.9)$$
> Effects: faster on **plateaus** (consecutive gradients with the same sign make the step grow), **damping** of the oscillations in directions whose sign keeps flipping (the canyon picture, i.e. a badly conditioned Hessian), and inertia that permits a larger $\eta$.
> **Nesterov:** apply the momentum first, $\boldsymbol{w}'=\boldsymbol{w}+\alpha\Delta\boldsymbol{w}_{old}$, evaluate the gradient **there**, then update $\boldsymbol{w}$. Improves the convergence rate in batch mode (not in the stochastic case).
> **The search-control view.** Backprop traces a **path** through the loss/weight space. The path depends on the data, the model, the starting point ([[Weight Initialization]] - small random weights, in the linear zone, symmetry broken), the speed ($\eta$, momentum), and where you stop ([[Stopping Criteria]], [[Early Stopping]]). Together they *are* the search bias.
> Consequence for [[Multiple Minima]]: the loss is non-convex, different inits give different models. This is not a disaster - report mean $\pm$ variance over trials, or keep all the runs as an [[Ensemble Learning|ensemble]]. If you pick one run, that is a **model selection** decision and needs a VL set.
> And the key reminder: we do **not** want the global minimizer of $R_{emp}$ - that is an overfitting solution. Optimization is not the goal.

### E9 [theory] Regularization vs early stopping

Both control complexity. Compare them: mechanism, cost, and the frequent misunderstanding about each.

> [!success]- Answer
> **Tikhonov / L2:** $\operatorname{Loss}(\boldsymbol{w})=\sum_p(d_p-o(\boldsymbol{x}_p))^2+\lambda\lVert\boldsymbol{w}\rVert^2$, implemented as weight decay $w_{new}=w+\eta\Delta w-2\lambda w$. Principled, always applicable, one number, and it lowers the effective VC directly (a NN's capacity depends on the number of weights **and their size**). Typical $\lambda$ is small, e.g. $0.01$, and it is **always** chosen by model selection.
> **Early stopping:** halt before $R_{emp}$ is fully minimized, because the effective number of parameters grows during training. Empirical, and it **spends validation data**; it gives no hyperparameter value, and it becomes awkward inside CV (a different stop point per fold - best fix: average the *training error* at each fold's validation optimum and retrain to that level of fitting, avoiding undertraining).
> **Misunderstanding 1:** regularization is *not* a device to stabilize training convergence; it controls **complexity**.
> **Misunderstanding 2:** ES is not an alternative that removes the need for validation - it *is* a model-selection decision, and it must never be based on the test set.
> A well-regularized model makes the VL curve follow the TR curve, so ES becomes unnecessary and ordinary convergence criteria suffice. You can use both: ES simply never fires if the VL error never rises.
> Other controls: **L1/lasso** (drives weights exactly to zero $\to$ feature selection, but not differentiable), elastic nets, data augmentation, [[Dropout]], [[Batch Normalization]], number of hidden units, and **architectural** regularization (weight sharing in [[CNN]]/[[RNN]]).

---

## Part F - Architectures and deep learning

### F1 [theory] CNN: what the two ingredients buy

Name the two architectural constraints of a convolutional layer and derive from them: translation invariance, parameter reduction, and the growth of the receptive field. What does pooling add?

> [!success]- Answer
> A convolutional layer applies one small kernel at **every position**: $S(i,j)=\sum_m\sum_n\boldsymbol{I}(i+m,j+n)\boldsymbol{K}(m,n)$; the output $S$ is a **feature map** - what that one filter found, everywhere. (Technically cross-correlation; since the kernel is learned, the flip is irrelevant.)
> **1. Local connections** - each unit sees only its receptive field, so it extracts *local* features. **2. Weight sharing** - the same kernel at every position.
> Consequences: the same feature is detected wherever it appears (**translation equivariance**, and invariance after pooling); far **fewer free parameters** with the same connectivity - *regularization by architecture*, exactly like the weight sharing over time of an RNN; and stacking layers makes the **receptive field grow with depth**, so global features are built from local ones.
> **Pooling** (max, mean, or subsampling by stride) reduces the map and makes the representation approximately **invariant to small translations**: a max changes little if a few inputs shift.
> Motivation: on $16\times16$ character images a fully connected net misses precisely because it lacks invariance to shifts and rotations.

### F2 [exercise] Feature-map size and parameter count

Input $32\times32\times3$; convolutional layer with 16 kernels of size $5\times5$, padding $p=2$.
(a) Output size with stride $s=1$? (b) With $s=2$? (c) Number of free parameters (with bias)? (d) Compare with a fully connected layer producing the same number of outputs as in (a).

> [!success]- Answer
> $o=\left\lfloor\frac{i-k+2p}{s}\right\rfloor+1$.
> (a) $\lfloor(32-5+4)/1\rfloor+1=31+1=\mathbf{32}$ $\to$ $32\times32\times16$ ("same" padding).
> (b) $\lfloor31/2\rfloor+1=15+1=\mathbf{16}$ $\to$ $16\times16\times16$: stride 2 subsamples.
> (c) Each kernel spans all 3 input channels: $(5\cdot5\cdot3+1)\times16=76\times16=\mathbf{1216}$ parameters.
> (d) Fully connected from $32\cdot32\cdot3=3072$ inputs to $32\cdot32\cdot16=16384$ outputs: $\approx3072\times16384\approx5.03\times10^{7}$ weights - about **41 000 times more**. Same connectivity pattern, incomparably fewer free parameters: that is what "architectural regularization" means concretely.

### F3 [theory] RNN

Write an RNN as a state transition system. State the three assumptions and what each one buys. How is it trained?

> [!success]- Answer
> $$\boldsymbol{x}(t)=\tau\big(\boldsymbol{x}(t-1),\boldsymbol{l}(t)\big),\qquad \boldsymbol{y}(t)=g\big(\boldsymbol{x}(t),\boldsymbol{l}(t)\big),\qquad \boldsymbol{x}(0)=\boldsymbol{0}$$
> and for a single recurrent unit $x(t)=f(w^Tl(t)+\hat w\,x(t-1)+\theta)$, where $x(t)$ is the **state** - a continuous-valued summary of the past - and $\hat w$ the recurrent weight.
>
> | assumption | what it buys |
> |---|---|
> | **causality** - the output at $t_0$ depends only on inputs at $t<t_0$ | an internal state is possible at all |
> | **stationarity** - $\tau$ is the same at every $t$ | one **fixed-size** model handles sequences of any length |
> | **adaptivity** - $\tau$ is a NN with free weights | the encoding of the past is *learnt*, not designed |
>
> **Training:** unfold through time into a feedforward network with **shared weights** and run backprop on it - BPTT (or RTRL for the same gradient computed on-line). Long dependencies suffer from vanishing/exploding gradients (gradient clipping, gated units).
> Used whenever the output depends on the **history** of the inputs, or the domain consists of variable-length sequences: signal processing and control, speech and NLP, time series, genomics, activity recognition.

### F4 [theory] Randomized NN and Echo State Networks

Describe the architecture and how it is trained. State the **Echo State Property** and the condition that enforces it. What is the *architectural bias* of a reservoir?

> [!success]- Answer
> **Architecture:** input $\to$ **untrained** hidden layer (random weights, many units) $\to$ **trained linear readout**. The hidden layer is a randomized, large [[Linear Basis Expansion]] justified by [[Cover's Theorem]]; only the readout is fitted, and it is a linear model:
> $$\boldsymbol{W}^{out}=\boldsymbol{H}^{+}\boldsymbol{d}\quad\text{or, regularized,}\quad \boldsymbol{W}^{out}=(\boldsymbol{H}^T\boldsymbol{H}+\lambda\boldsymbol{I})^{-1}\boldsymbol{H}^T\boldsymbol{d},\qquad\boldsymbol{H}=f(\boldsymbol{W}\boldsymbol{X})$$
> i.e. pseudoinverse / ridge regression - closed form, no learning rate, no local minima. Named families: **ELM**, **RVFL** (which keeps direct input-to-readout links).
> **ESN / Reservoir Computing** applies this to recurrent nets: a large, **sparsely connected** untrained recurrent reservoir plus a linear readout.
> **Echo State Property:** contractivity of the state transition function, obtained by limiting the **spectral radius** $\rho$ of the reservoir matrix. Consequence: the state asymptotically depends only on the input history and **not on the initial state**.
> **Architectural bias:** even untrained, the reservoir discriminates sequences in a **suffix-based** way - states are close in proportion to the length of the common suffix. Markovian tasks are therefore easy, and this bias is present with full training too.
> Why it matters: the cost of BPTT disappears $\to$ big data, embedded/edge devices, photonic hardware. **DeepESN** adds layering, which yields multiple time-scales (higher layers focus on lower frequencies). The same idea moves from time to graphs in [[Graph Echo State Networks]].

### F5 [theory] Why depth?

Universal approximation says one hidden layer is enough. Explain, with the parity example, what depth actually buys, and what "no-flattening" means.

> [!success]- Answer
> The gain is **efficiency, not expressivity**.
> **Parity** (output 1 iff an odd number of the $N$ binary inputs are 1, i.e. XOR generalized):
> - depth 2 (disjunctive normal form: AND gates then one OR) needs one AND per positive configuration $\to$ $2^{N-1}+1$ gates - **exponential** (8 inputs: 129 gates);
> - $\log N$ layers (a binary tree of XOR nodes, $N-1$ internal nodes, 3 gates each) $\to$ $3(N-1)$ gates - **polynomial** (8 inputs: 21 gates).
>
> **No-flattening results:** there are families of functions approximable efficiently by an architecture of depth $>d$ that need a **much larger (up to exponentially larger)** model when the depth is limited to $\le d$. Which families exactly is still open.
> **Why:** **compositionality**. Concepts learned in one layer are reused as primitives by the next, so the model never has to represent all the combinations explicitly - you learn the sub-patterns, not every combination of them. Depth is therefore itself an **inductive bias**: *the target is a composition of simpler functions*. When that assumption fits you get an exponential saving; when it does not, [[No Free Lunch]] still applies.
> Depth also attacks the [[Curse of Dimensionality]] by allowing **non-local generalization**: a composition-of-factors bias distinguishes exponentially many regions with $O(k)$ examples ([[Distributed Representation]]).

### F6 [theory] Deep Graph Networks

Write the message-passing equation and explain the three consequences. Why can a CNN not simply be applied to a graph?

> [!success]- Answer
> For $l=1,\dots,L$ and every node $v$:
> $$\boldsymbol{h}_v^{(l)}=\mathrm{AGG}_{\boldsymbol{W}^{(l)}}\Big(\boldsymbol{L}_v,\ \boldsymbol{h}_v^{(l-1)},\ \{\boldsymbol{h}_u^{(l-1)}:u\in\mathcal{N}(v)\}\Big)$$
> with $\boldsymbol{L}_v$ the node label, $\mathcal{N}(v)$ read from the adjacency matrix, $\mathrm{AGG}$ a **permutation-invariant** operator (sum, mean, max, attention) and $\boldsymbol{W}^{(l)}$ shared across all nodes.
> 1. **Permutation invariance** follows from $\mathrm{AGG}$ being order-blind plus weight sharing across nodes: the model abstracts from any ordering of the vertices.
> 2. **After $l$ steps the state of $v$ covers a context of radius $l$** - so depth is not a tuning choice, it is *how information reaches distant nodes*. One step only sees the immediate neighbours.
> 3. **Two ways to iterate:** each step a new layer with its own weights (*layering/convolutional*: GCN, GAT, NN4G) or the same function iterated to a fixed point (*recursive*: GNN, GraphESN).
>
> **Why not a CNN.** A CNN kernel lives on a **regular grid** where the neighbours of a pixel are ordered and fixed in number; the neighbours of a vertex are **unordered and variable in size**, so a fixed-shape kernel cannot be placed on them. The fix keeps the two CNN ingredients and drops the grid: constrain the receptive field to $\mathcal{N}(v)$, share weights over all vertices, and extend beyond the local neighbourhood through **layering**.
> Graph-level outputs need a permutation-invariant **global pooling** before the readout; node-level tasks emit one output per node. The motto: *move the models to the data, not the data to the models*.

---

## Part G - Learning without a teacher

### G1 [derivation + exercise] K-means

Write the vector-quantization loss, derive the on-line update rule from it, and state the batch algorithm. Then run it on $\{1,2,3,10,11,12\}$ with initial centres $w_1=1$, $w_2=2$.

> [!success]- Answer
> **Loss:** $E=\sum_i\sum_j\lVert\boldsymbol{x}_i-\boldsymbol{w}_j\rVert^2\,\text{winner}(i,j)$, where winner$(i,j)=1$ iff $j$ is the closest prototype to $\boldsymbol{x}_i$.
> **On-line rule:** differentiating w.r.t. $\boldsymbol{w}_j$ (only the winner's term survives) gives
> $$\Delta\boldsymbol{w}_{i^*}=\eta\,(\boldsymbol{x}_i-\boldsymbol{w}_{i^*})$$
> - only the **winner** moves, and it moves towards the pattern. This is stochastic gradient descent: local minima and dependence on the initialization.
> **Batch algorithm:** (1) choose $K$ centres; (2) **assign** each pattern to the nearest centre, $i^*(\boldsymbol{x})=\arg\min_i\lVert\boldsymbol{x}-\boldsymbol{w}_i\rVert^2$; (3) **recompute** each centre as the mean of its members; (4) repeat until (almost) no reassignment or negligible decrease of $E$.
>
> **Run.**
> - *Iter 1.* Assign: $1\to w_1$; $2,3,10,11,12\to w_2$ (3 is at distance 2 from $w_1$ and 1 from $w_2$). New centres: $w_1=1$, $w_2=38/5=7.6$.
> - *Iter 2.* Assign: $1,2,3\to w_1$ (distances $0,1,2$ vs $6.6,5.6,4.6$); $10,11,12\to w_2$. New centres: $w_1=2$, $w_2=11$.
> - *Iter 3.* No reassignment $\to$ **converged**: clusters $\{1,2,3\}$ and $\{10,11,12\}$, centres $2$ and $11$.
>
> Note the first iteration produced a badly unbalanced partition purely because of the initialization; a worse init can leave it stuck. Hence: multiple restarts, $K$ chosen outside the algorithm, and remember that with no target **the metric is the whole inductive bias** and evaluating the result is largely subjective.

### G2 [theory] Self-Organizing Maps

Write the two rules of a SOM, explain what makes it *topology preserving*, and say what it reduces to if the neighbourhood function is removed.

> [!success]- Answer
> For each drawn input $\boldsymbol{x}$ at iteration $t$:
> $$\textbf{compete:}\quad i^*(\boldsymbol{x})=\arg\min_i\lVert\boldsymbol{x}-\boldsymbol{w}_i\rVert$$
> $$\textbf{cooperate:}\quad \Delta\boldsymbol{w}_i=\eta(t)\,h_{i,i^*(\boldsymbol{x})}(t)\,(\boldsymbol{x}-\boldsymbol{w}_i)\qquad\forall i$$
> $N$ units sit on a low-dimensional (usually 2D) **lattice**; $i$ is a *coordinate on the map*, not a data point. $h_{i,j}(t)$ is a neighbourhood function - a gaussian of width $\sigma(t)$ - decreasing in $\lVert i-j\rVert$, the distance **between unit coordinates on the lattice**, never between weight vectors. Both $\eta(t)$ and $\sigma(t)$ shrink with $t$ (ordering phase, then convergence phase).
> **Topology preservation:** because the winner drags its lattice neighbours along, neighbouring units end up responding to similar inputs, and inputs close in the input space map onto the same or nearby units. That is what turns a set of prototypes into an ordered **map**.
> Set $h_{i,j}=\delta_{ij}$ (winner only) and the algorithm collapses to on-line [[K-Means]] - the neighbourhood term *is* the SOM.
> Uses: clustering, vector quantization / pattern recognition, data compression (transmit the winner's index), and 2D visualization of a multidimensional distribution.

---

## Part H - Validation

### H1 [theory] Model selection vs model assessment

Define the two goals, assign the three data sets, and state the gold rule. What is the development set?

> [!success]- Answer
> - **Model selection:** compare candidate models / hyperparameter configurations $\Lambda$ (#units, $\lambda$, $\eta$, $k$, kernel, $C$, ...) and choose. **It returns a model.**
> - **Model assessment:** estimate the prediction error of the chosen model on new data. **It returns an estimation value.**
>
> | set | role |
> |---|---|
> | $D_{TR}$ | fit the parameters (minimize $R_{emp}$) - training |
> | $D_{VL}$ | select model and hyperparameters - model selection |
> | $D_{TS}$ | final risk estimate - model assessment |
>
> All disjoint; e.g. 50/25/25. $D_{TR}\cup D_{VL}$ is the **development / design set**: everything used to build the final model.
> **Gold rule:** keep the goals separate and use separate data in each phase. Two corollaries: the TR result is not a good estimate of VL/TS, and the VL result is not a good estimate of TS. Hence: never report VL error as a risk estimate, and never use TS results for selection **in any form** - the moment a set influences a design choice it stops being an unbiased test set. (Like a test with the answer key next to it.)

### H2 [theory] K-fold cross validation

Describe the procedure. When does CV return a *model* and when only an *estimate*? What is nested CV for?

> [!success]- Answer
> Split $D$ into $K$ mutually exclusive folds; for each $i$, train on $D\setminus D_i$ and test on $D_i$; aggregate. All the data are used both for training and for testing, and you get a **standard deviation over the folds** - report MEAN $\pm$ SD.
> - Used for **model selection**: it *does* return a final model - after picking the best $\Lambda$, **retrain on the whole data set**.
> - Used for **assessment**: it does *not* give a unique model (one per fold); what you take away is a risk estimate with its variability.
>
> **Nested / double CV:** an **inner** CV does model/hyperparameter selection, an **outer** CV assesses the whole selection *procedure* - a new selection is performed inside every outer fold.
> Variants: hold-out (cheap, split-dependent); LOO ($K=l$: almost no data wasted, expensive, high variance of the estimate); **5- or 10-fold** as the recommended compromise; repeated CV (average several random splits, especially for *comparing* models); [[Stratified Sampling]] to keep class proportions; [[Bootstrap]].
> And: the best model is chosen by looking at the **validation** result, never by mixing in the training error.

### H3 [exercise] Cost of a nested CV

You use a 5-fold outer CV for assessment, a 4-fold inner CV for selection, and a grid of 12 hyperparameter configurations. How many model trainings does the whole procedure require, including the final model delivered to the user?

> [!success]- Answer
> Inner CV for one outer fold: $4\times12=48$ trainings. Then 1 retraining on that outer fold's full development set to get the model to be tested: $48+1=49$.
> Over 5 outer folds: $5\times49=245$.
> Plus 1 final retraining on **all** the data, with the configuration chosen from the whole procedure: $\mathbf{246}$ trainings.
> Two comments: the cost is multiplicative, which is why grid granularity and $K$ are real design decisions; and most of a project's running time lives here, not in a single training run. Be reasonable - avoid both a too rough and a never-ending evaluation.

### H4 [theory] The counterexample: why assessment must be done right

Describe the 1000-random-features experiment, name the effect, and list the three ways to avoid it.

> [!success]- Answer
> **Setup:** 20-30 examples, 1000 random input variables, a **random** 0/1 target. Screen all 1000 single-feature models and keep the best; on any subsequent TR/VL/TS split it scores 99-100%.
> **Reality:** the correct test error is **50%** - a coin. The target was random; there is nothing to learn.
> **What went wrong:** the selection used *all* the data, so the test set was implicitly consumed at the very beginning. This is **subset selection bias**. K-fold CV does not fix it by itself - it can even be more convincing and therefore more misleading.
> **Fixes:**
> 1. Separate the test set **in advance**, before any model *or feature* selection, and check the winner there.
> 2. For selection with CV: use "CV + external test", or **double CV** with a fresh selection inside each fold.
> 3. For assessment with CV: do the **feature selection inside each fold** - then the random task correctly yields $\approx50\%$.
>
> This is about the **correctness of the estimate**, not about whether the task is solvable. Related traps: selecting/scaling features before splitting; reporting the validation number as a test number; comparing models on a single split; tuning hyperparameters **sequentially** rather than on a joint grid (an $\eta$ chosen with 10 units need not suit 100 units).

### H5 [exercise] Confusion matrix

A classifier on 500 patients gives $TP=40$, $FN=10$, $FP=20$, $TN=430$. Compute accuracy, recall, specificity, precision and F1. Then compute the accuracy of the trivial "always negative" classifier and comment.

> [!success]- Answer
> - Accuracy $=\frac{TP+TN}{500}=\frac{470}{500}=\mathbf{0.94}$
> - Recall / sensitivity / TP rate $=\frac{TP}{TP+FN}=\frac{40}{50}=\mathbf{0.80}$
> - Specificity / TN rate $=\frac{TN}{TN+FP}=\frac{430}{450}=\mathbf{0.956}$ (FP rate $=0.044$)
> - Precision $=\frac{TP}{TP+FP}=\frac{40}{60}=\mathbf{0.667}$
> - F1 $=\frac{2\cdot0.667\cdot0.80}{0.667+0.80}=\mathbf{0.727}$
>
> **Trivial classifier** (predict "negative" always): $TN=450$, accuracy $=450/500=\mathbf{0.90}$ - only 4 points below a classifier that actually finds 80% of the cases, and with recall $0$.
> Moral: with **unbalanced** data accuracy alone is meaningless. Report recall/precision (or the confusion matrix itself), and remember that for a *balanced* binary task 50% accuracy is a coin.

### H6 [exercise] ROC and AUC

Scores and true labels: $0.9(+),\ 0.8(-),\ 0.7(+),\ 0.6(+),\ 0.4(-),\ 0.2(-)$. Compute the AUC by counting pairs. What does the ROC curve give you that a single accuracy figure does not?

> [!success]- Answer
> AUC = fraction of (positive, negative) pairs ranked correctly. Positives $\{0.9,0.7,0.6\}$, negatives $\{0.8,0.4,0.2\}$, $3\times3=9$ pairs.
> - $0.9>0.8,0.4,0.2$ $\to$ 3
> - $0.7>0.4,0.2$ (but $0.7<0.8$) $\to$ 2
> - $0.6>0.4,0.2$ $\to$ 2
>
> AUC $=7/9\approx\mathbf{0.78}$.
> The ROC plots TP rate against FP rate ($1-$ specificity) as the decision threshold sweeps the score range (e.g. the 0.5 of a logistic unit). The diagonal is random guessing; better curves lie above it. It is **threshold-independent**, so it summarizes the classifier rather than one operating point - and it is the right tool for *choosing* the threshold by studying the FP/FN trade-off (a rejection zone around the threshold is also an option).

---

## Closing: the three things they will ask in some form

> [!important] Gold rule
> $D_{TR}$ trains; $D_{VL}$ selects; $D_{TS}$ estimates. Once a set has influenced a design choice it is no longer an unbiased test set. $\to$ [[Validation Pitfalls]]

> [!important] The trade-off
> Fitting capability *vs* model complexity, in two languages: $R\le R_{emp}+\epsilon(1/l,VC,1/\delta)$ and $\text{bias}^2+\text{variance}+\text{noise}$. $\to$ [[Statistical Learning Theory]], [[Bias-Variance Tradeoff]]

> [!important] Bias is necessary
> A learner with no [[Inductive Bias]] is a lookup table: it cannot generalize. Proved in [[Concept Learning]].
