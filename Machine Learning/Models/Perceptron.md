---
tags: [machine-learning, models, neural-networks]
---

# Perceptron

A single LTU trained to **minimize the number of misclassified patterns**: find $w$ s.t. $\operatorname{sign}(w^Tx)=d$.

Historically important: a biologically inspired model with well-defined computational capabilities, **proved** by a theorem.

## The Perceptron Learning Algorithm

1. Initialize the weights (zero or small random).
2. Pick a learning rate $\eta\in(0,1)$.
3. Until the stopping condition (e.g. the weights don't change), for each training pattern $(x,d)$ with $d=\pm1$:
   - compute $out=\operatorname{sign}(w^Tx)$;
   - if $out=d$, don't touch the weights;
   - if $out\neq d$, update

     $$
     w_{new}=w+\eta\,d\,x
     \qquad\Big(\text{i.e. } +\eta x \text{ if } w^Tx\le0,\ d=+1;\ \ -\eta x \text{ if } w^Tx>0,\ d=-1\Big).
     $$

Equivalent form: $w_{new}=w+\eta\frac12(d-out)x$.

It is an **on-line** algorithm: one step per pattern. Geometrically, $w$ moves a little towards a misclassified positive pattern - the Hebbian rule.

> [!note]
> The $\delta$ here is $(d-\operatorname{sign}(w^Tx))$, while in LMS it is $(d-w^Tx)$: LMS is derived by the gradient **without** the threshold, so it changes the weights also for correctly classified patterns and does not minimize misclassifications. See [[Neural Networks]] for the full comparison.

## Properties

- The perceptron can always **learn** what it is able to **represent** (linear decision boundaries, see [[Linear Models]]).
- It is **guaranteed to converge** in a finite number of steps if the problem is linearly separable, independently of the starting point (although the final solution is not unique and does depend on it).
- If the problem is not separable it may be unstable and develop cycles, with weights that are not necessarily optimal.

## Reduction to an "all positive" task

$$
\exists w^* ~|~ d_i(w^*x_i) \ge \alpha, \qquad \alpha = \min_i(d_i(w^*x_i)) > 0
$$

Assume $(x_i, d_i)$ in the TR set, with $d_i=+1$ or $d_i=-1$, $i=1..l$. Then

$$
d_i(w^*x_i) \ge \alpha \implies w^*(d_ix_i) \ge \alpha
$$

Defining $x_i'=(d_ix_i)$, then $w^* \text{ solution} \iff w^* \text{ solution of } (x', +1)$

$$
w^* \text{ solution} \implies d_i(w^*x_i) \ge \alpha \implies (w^*d_ix_i) \ge \alpha \implies (w^*x_i') \ge \alpha \implies w^* \text{ solution of } (x', +1)
$$
$$
 w^* \text{ solution of } (x', +1) \implies (w^*x_i') \ge \alpha \implies (w^*d_ix_i) \ge \alpha \implies d_i(w^*x_i) \ge \alpha \implies w^* \text{ solution}
$$

So we can work with the subset of misclassified patterns as if all $d_i=+1$.

## Perceptron Convergence Theorem

> For linearly separable tasks, the perceptron algorithm converges after a finite number of steps.

**Idea of the proof:** find a lower bound growing like $q^2$ and an upper bound growing like $q$ for $\lVert w(q)\rVert^2$; they can only both hold up to a finite $q$.

### Assumptions
- $w(0) = 0$
- $\eta = 1$
- Linearly separable problem

### Proof
After $q$ false negatives (predicting $-1$ on $d_i=+1$), with $x_i(j)$ the $j$-th misclassified pattern:
$$
w(q) = \sum_j^q x_i(j), \quad w(j) = w(j-1) + \eta~d_i~x_i(j) = w(j-1) + x_i(j)
$$
$$
w^* \text{ solution}, \qquad\alpha = \min_i ((w^*)^Tx_i), \quad \beta = \max_i||x_i||^2
$$

##### Lower Bound

Multiply both terms by $w^*$:
$$
w(q) = \sum_j^q x_i(j) \implies (w^*)^Tw(q) = (w^*)^T\sum_j^q x_{i}(j) \ge q\alpha
$$
For the **Cauchy-Schwarz** property ($(\alpha^T \beta)^2 \le ||\alpha||^2 ||\beta||^2$):
$$
||w^*||^2 ~ ||w(q)||^2 \ge ((w^*)^Tw(q))^2 \ge (q\alpha)^2
$$
We have the **lower bound**:
$$
||w(q)||^2 \ge \frac{(q\alpha)^2}{||w^*||^2}
$$

##### Upper Bound
For the property $||\alpha + \beta||^2 = \sum_i (\alpha_i + \beta_i)^2 = \sum_i (\alpha_i^2 + 2\alpha_i\beta_i + \beta_i^2) = ||\alpha||^2 + 2\alpha\beta + ||\beta||^2$:
$$
||w(q)||^2 = ||w(q-1) + x_i(q)||^2 = ||w(q-1)||^2 + 2w(q-1)x_i(q) + ||x_i(q)||^2
$$
For the *q-th error*, $2w(q-1)x_i(q) \le 0$ since it misclassifies $d_i=+1$, so:
$$
||w(q)||^2 \le ||w(q-1)||^2 + ||x_i(q)||^2
$$
So, considering $w(0)=0 \implies w(q) = \sum_i x(i)$, we have the **upper bound**:
$$
||w(q)||^2 \le \sum_j^q||x_i(j)||^2 \le q\beta
$$

##### Conclusion
Joining together the **upper bound** $q\beta$ and the **lower bound** $(q\alpha)^2/||w^*||^2$
$$
	q\beta \ge ||w(q)||^2 \ge (q\alpha)^2/||w^*||^2
$$
$$
	q\beta \ge q^2\alpha', \quad \alpha'=\alpha^2/||w^*||^2
$$
$$
\beta \ge q\alpha'
$$
So we have an upper bound on the number of $q$ steps (finite)
$$
q \le \beta/\alpha'
$$

## Related

- [[Neural Networks]] - perceptron vs LMS/Adaline, and from LTU to MLP.
- [[Delta Rule]], [[Linear Models]], [[VC Dimension]].
