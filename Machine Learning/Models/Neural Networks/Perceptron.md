---
tags: [machine-learning, models, algorithms]
---

# Perceptron

A single LTU trained to **minimize the number of misclassified patterns**: find $\boldsymbol{w}$ s.t. $\operatorname{sign}(\boldsymbol{w}^T\boldsymbol{x})=d$.

## The Perceptron Learning Algorithm

1. Initialize the weights (zero or small random).
2. Pick a learning rate $\eta\in(0,1)$.
3. Until the stopping condition (e.g. the weights don't change), for each training pattern $(\boldsymbol{x},d)$ with $d=\pm1$:
   - compute $out=\operatorname{sign}(\boldsymbol{w}^T\boldsymbol{x})$;
   - if $out=d$, don't touch the weights;
   - if $out\neq d$, update

     $$
     \boldsymbol{w}_{new}=\boldsymbol{w}+\eta\,d\,\boldsymbol{x}
     \qquad\Big(\text{i.e. } +\eta\boldsymbol{x} \text{ if } \boldsymbol{w}^T\boldsymbol{x}\le0,\ d=+1;\ \ -\eta\boldsymbol{x} \text{ if } \boldsymbol{w}^T\boldsymbol{x}>0,\ d=-1\Big).
     $$

Equivalent form: $\boldsymbol{w}_{new}=\boldsymbol{w}+\eta\frac12(d-out)\boldsymbol{x}$.

It is an **on-line** algorithm: one step per pattern. Geometrically, $\boldsymbol{w}$ moves a little towards a misclassified positive pattern - the Hebbian rule.

> [!note]
> The $\delta$ here is $(d-\operatorname{sign}(\boldsymbol{w}^T\boldsymbol{x}))$, while in LMS it is $(d-\boldsymbol{w}^T\boldsymbol{x})$: LMS is derived by the gradient **without** the threshold, so it changes the weights also for correctly classified patterns and does not minimize misclassifications. See [[Neural Networks]] for the full comparison.

## Properties

- The perceptron can always **learn** what it is able to **represent** (linear decision boundaries, see [[Linear Models]]).
- It is **guaranteed to converge** in a finite number of steps if the problem is linearly separable, independently of the starting point (although the final solution is not unique and does depend on it).
- If the problem is not separable it may be unstable and develop cycles, with weights that are not necessarily optimal.

## Reduction to an "all positive" task

$$
\exists \boldsymbol{w}^* ~|~ d_i(\boldsymbol{w}^{*T}\boldsymbol{x}_i) \ge \alpha, \qquad \alpha = \min_i\big(d_i(\boldsymbol{w}^{*T}\boldsymbol{x}_i)\big) > 0
$$

Assume $(\boldsymbol{x}_i, d_i)$ in the TR set, with $d_i=+1$ or $d_i=-1$, $i=1..l$. Then

$$
d_i(\boldsymbol{w}^{*T}\boldsymbol{x}_i) \ge \alpha \implies \boldsymbol{w}^{*T}(d_i\boldsymbol{x}_i) \ge \alpha
$$

Defining $\boldsymbol{x}_i'=d_i\boldsymbol{x}_i$, then $\boldsymbol{w}^* \text{ solution} \iff \boldsymbol{w}^* \text{ solution of } (\boldsymbol{x}', +1)$:

$$
\boldsymbol{w}^* \text{ solution} \implies d_i(\boldsymbol{w}^{*T}\boldsymbol{x}_i) \ge \alpha \implies \boldsymbol{w}^{*T}(d_i\boldsymbol{x}_i) \ge \alpha \implies \boldsymbol{w}^{*T}\boldsymbol{x}_i' \ge \alpha
$$
$$
\boldsymbol{w}^* \text{ solution of } (\boldsymbol{x}', +1) \implies \boldsymbol{w}^{*T}\boldsymbol{x}_i' \ge \alpha \implies \boldsymbol{w}^{*T}(d_i\boldsymbol{x}_i) \ge \alpha \implies d_i(\boldsymbol{w}^{*T}\boldsymbol{x}_i) \ge \alpha
$$

So we can work with the subset of misclassified patterns as if all $d_i=+1$, and **every mistake is a false negative** ($\boldsymbol{w}^T\boldsymbol{x}_i\le0$ while the target is $+1$). That case is therefore fully general.

## Perceptron Convergence Theorem

> For linearly separable tasks, the perceptron algorithm converges after a finite number of steps.

### Assumptions

- $\boldsymbol{w}(0) = \boldsymbol{0}$
- $\eta = 1$
- Linearly separable problem, so some $\boldsymbol{w}^*$ exists with $\boldsymbol{w}^{*T}\boldsymbol{x}_i>0$ for every pattern.

### What is actually being counted

> [!important]
> $q$ counts **mistakes** - i.e. weight updates - not patterns seen and not epochs run. The theorem bounds how many times the algorithm can ever correct itself.

After $q$ mistakes, with $\boldsymbol{x}_i(j)$ the $j$-th misclassified pattern:

$$
\boldsymbol{w}(j) = \boldsymbol{w}(j-1) + \eta\,d_i\,\boldsymbol{x}_i(j) = \boldsymbol{w}(j-1) + \boldsymbol{x}_i(j)
\qquad\Longrightarrow\qquad
\boldsymbol{w}(q) = \sum_{j=1}^{q} \boldsymbol{x}_i(j)
$$

### The two constants

$$
\alpha = \min_i \big(\boldsymbol{w}^{*T}\boldsymbol{x}_i\big), \qquad \beta = \max_i\lVert\boldsymbol{x}_i\rVert^2
$$

**$\alpha$ - the margin of the solution.** Take the solution vector $\boldsymbol{w}^*$ and project every training pattern onto it. All the projections are positive - that is what "solution" means. $\alpha$ is the **smallest** one: the pattern sitting closest to the separating hyperplane. It is strictly positive because the set is finite and separable.

So $\alpha$ measures **how comfortably the data is separated**. A wide street between the classes gives a large $\alpha$; classes that almost touch give a tiny one.

**$\beta$ - the radius of the data.** The squared length of the longest input vector: every pattern lives inside a ball of radius $\sqrt\beta$. This caps how much damage a single update can do, since an update adds exactly $\boldsymbol{x}_i$ to $\boldsymbol{w}$.

### Idea of the proof

Find a lower bound growing like $q^2$ and an upper bound growing like $q$ for $\lVert\boldsymbol{w}(q)\rVert^2$. They can only both hold up to a finite $q$.

---

#### Lower bound - about **alignment**

Every mistake adds $\boldsymbol{x}_i$ to $\boldsymbol{w}$, and every $\boldsymbol{x}_i$ has a component of **at least $\alpha$** along $\boldsymbol{w}^*$. So the projection grows by at least $\alpha$ per mistake: steady, guaranteed progress towards the solution direction, never undone.

$$
\boldsymbol{w}(q) = \sum_{j=1}^{q} \boldsymbol{x}_i(j) \implies \boldsymbol{w}^{*T}\boldsymbol{w}(q) = \boldsymbol{w}^{*T}\sum_{j=1}^{q} \boldsymbol{x}_{i}(j) \ge q\alpha
$$

By **Cauchy-Schwarz**, $(\boldsymbol{a}^T\boldsymbol{b})^2 \le \lVert\boldsymbol{a}\rVert^2\lVert\boldsymbol{b}\rVert^2$ - a projection can never exceed the vector's own length - so the *length* of $\boldsymbol{w}$ is forced to grow too:

$$
\lVert\boldsymbol{w}^*\rVert^2\,\lVert\boldsymbol{w}(q)\rVert^2 \ge \big(\boldsymbol{w}^{*T}\boldsymbol{w}(q)\big)^2 \ge (q\alpha)^2
$$

$$
\boxed{\ \lVert\boldsymbol{w}(q)\rVert^2 \ge \frac{(q\alpha)^2}{\lVert\boldsymbol{w}^*\rVert^2}\ }
$$

#### Upper bound - about **length**

Using $\lVert\boldsymbol{a}+\boldsymbol{b}\rVert^2 = \lVert\boldsymbol{a}\rVert^2 + 2\boldsymbol{a}^T\boldsymbol{b} + \lVert\boldsymbol{b}\rVert^2$:

$$
\lVert\boldsymbol{w}(q)\rVert^2 = \lVert\boldsymbol{w}(q-1) + \boldsymbol{x}_i(q)\rVert^2 = \lVert\boldsymbol{w}(q-1)\rVert^2 + 2\,\boldsymbol{w}(q-1)^T\boldsymbol{x}_i(q) + \lVert\boldsymbol{x}_i(q)\rVert^2
$$

The crucial step is the cross term: $2\,\boldsymbol{w}(q-1)^T\boldsymbol{x}_i(q) \le 0$ **precisely because $\boldsymbol{x}_i(q)$ was misclassified**. The very fact that we made a mistake means the new pattern points "away" from the current $\boldsymbol{w}$, so adding it cannot inflate the norm much. Dropping the negative term:

$$
\lVert\boldsymbol{w}(q)\rVert^2 \le \lVert\boldsymbol{w}(q-1)\rVert^2 + \lVert\boldsymbol{x}_i(q)\rVert^2
$$

Iterating from $\boldsymbol{w}(0)=\boldsymbol{0}$:

$$
\boxed{\ \lVert\boldsymbol{w}(q)\rVert^2 \le \sum_{j=1}^{q}\lVert\boldsymbol{x}_i(j)\rVert^2 \le q\beta\ }
$$

> [!important] Why the two bounds fight each other
> The **same event** - a mistake - buys a fixed amount of **alignment** while costing only a bounded amount of **length**. Alignment grows like $q$, length like $\sqrt q$. A vector cannot be more aligned with $\boldsymbol{w}^*$ than its own length allows (the cosine is at most 1), so this cannot go on forever.

#### The squeeze

$$
q\beta \;\ge\; \lVert\boldsymbol{w}(q)\rVert^2 \;\ge\; \frac{q^2\alpha^2}{\lVert\boldsymbol{w}^*\rVert^2}
$$

A **line** and a **parabola**, both starting at zero. The parabola overtakes the line at

$$
q\beta \ge q^2\alpha', \quad \alpha'=\frac{\alpha^2}{\lVert\boldsymbol{w}^*\rVert^2}
\qquad\Longrightarrow\qquad
\beta \ge q\alpha'
\qquad\Longrightarrow\qquad
q \le \frac{\beta}{\alpha'}
$$

Past that point the "lower bound" would sit **above** the "upper bound", which is impossible. So no such $q$ can be reached: the algorithm has run out of mistakes to make, and it has converged.

---

### Reading the bound

$\alpha$ on its own is **not** a meaningful quantity, because it is not scale-invariant: rescale $\boldsymbol{w}^*$ to $2\boldsymbol{w}^*$ and $\alpha$ doubles, even though the decision boundary is identical (the scaling freedom of [[Linear Models]]). That is exactly why the final bound divides by $\lVert\boldsymbol{w}^*\rVert^2$:

$$
q \le \frac{\beta\,\lVert\boldsymbol{w}^*\rVert^2}{\alpha^2}
$$

The ratio $\gamma=\alpha/\lVert\boldsymbol{w}^*\rVert$ **is** scale-invariant: it is the geometric distance from the closest pattern to the hyperplane. Writing $R=\sqrt\beta$ for the data radius:

$$
\boxed{\ q \le \left(\frac{R}{\gamma}\right)^2\ }
\qquad\text{i.e.}\qquad
\left(\frac{\text{size of the data}}{\text{width of the gap}}\right)^2
$$

Notice what is **absent**: the number of patterns and the dimensionality. Only the geometry matters.

Two things to carry away:

- The bound holds for **any** valid $\boldsymbol{w}^*$, so the tightest version comes from the one maximizing $\gamma$ - which is precisely the maximum-margin separator, and precisely the idea [[SVM]] is built on.
- $\gamma$ appears **squared in the denominator**, so a barely separable problem can take enormously many steps. Convergence is guaranteed, but "finite" can still mean impractical.

## Related

- [[Neural Networks]] - perceptron vs LMS/Adaline, and from LTU to MLP.
- [[SVM]] - what you get by maximizing $\gamma$ instead of just reaching $\gamma>0$.
- [[Delta Rule]], [[Linear Models]], [[VC Dimension]].
