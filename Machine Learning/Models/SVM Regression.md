---
tags: [machine-learning, models]
---

# SVM Regression

Support Vector Regression (SVR): the same machinery as [[SVM]], with a loss that ignores small errors.

## Setup

$$
d=f(x)+v
$$

with $f$ unknown and the noise $v$ statistically independent of $x$. Given $T=\{(x_i,d_i)\}_{i=1}^{N}$, estimate $d$ with a linear expansion of non-linear functions ([[Linear Basis Expansion]]):

$$
y=h(x)=w^T\Phi(x),\qquad
\Phi(x)=(\phi_0(x)=1,\phi_1(x),\dots,\phi_{m_1}(x))^T,\quad w(0)=b .
$$

## $\epsilon$-insensitive loss

$$
L_\epsilon(d,y)=
\begin{cases}
\lvert d-y\rvert-\epsilon & \text{if } \lvert d-y\rvert\ge\epsilon\\
0 & \text{otherwise}
\end{cases}
$$

Also called the *soft margin loss*: a flat bottom of width $2\epsilon$, then linear growth.

### The $\epsilon$-tube

The loss draws a tube of half-width $\epsilon$ around the model prediction. Points **inside** the tube cost nothing.

> [!important]
> Only the support vectors guide the solution, because the cost function **ignores any training data close to the model prediction**. Exactly the same role as in classification, where the ignored points are the ones lying beyond the margin.

## Slack variables

Two per pattern now, one for each side of the tube:

$$
-\xi_i'-\epsilon \le d_i-w^T\Phi(x_i) \le \epsilon+\xi_i
$$

giving the constraints

$$
\begin{cases}
d_i-w^T\Phi(x_i)\le \epsilon+\xi_i\\
w^T\Phi(x_i)-d_i\le \epsilon+\xi_i'\\
\xi_i\ge0,\quad \xi_i'\ge0
\end{cases}
\qquad \forall i=1,\dots,N
\qquad (\xi_i \text{ above the tube},\ \xi_i' \text{ below})
$$

## Primal problem

$$
\Psi(w,\xi,\xi')=\tfrac12 w^Tw + C\sum_{i=1}^{N}(\xi_i+\xi_i')
$$

under the constraints above. Same shape as the soft-margin classifier, with the two slacks summed.

## Dual problem

$$
Q(\alpha,\alpha')=\sum_{i=1}^{N}d_i(\alpha_i-\alpha_i')-\epsilon\sum_{i=1}^{N}(\alpha_i+\alpha_i')-\frac12\sum_{i,j=1}^{N}(\alpha_i-\alpha_i')(\alpha_j-\alpha_j')\,k(x_i,x_j)
$$

subject to

$$
\sum_{i=1}^{N}(\alpha_i-\alpha_i')=0,
\qquad
0\le\alpha_i\le C,\quad 0\le\alpha_i'\le C .
$$

The [[Kernel Methods|kernel]] appears in exactly the same place as in classification.

## Weight vector and estimate

Defining $\gamma_i=\alpha_i-\alpha_i'$:

$$
w=\sum_{i=1}^{N}(\alpha_i-\alpha_i')\Phi(x_i)=\sum_{i=1}^{N}\gamma_i\Phi(x_i)
$$

> [!note] Support vector
> Here the support vectors are the patterns with **non-zero $\gamma_i$**.

And the estimate needs no $\Phi$ at all:

$$
h(x)=w^T\Phi(x)=\sum_{i=1}^{N}\gamma_i\,\Phi^T(x_i)\Phi(x)=\sum_{i=1}^{N}\gamma_i\,k(x_i,x)
$$

a linear combiner of dot products in a feature space we can ignore.

## Recipe

1. Choose the user-specified parameters $C$ **and $\epsilon$**.
2. Choose the inner-product kernel $k$.
3. Compute the kernel matrix $K$.
4. Solve the dual (QP) $\to$ $\{\gamma_i\}$.
5. Compute the bias.
6. Predict with $h(x)=\sum_i\gamma_i k(x_i,x)$.

$C$, $\epsilon$ and the kernel parameters are all hyperparameters $\to$ [[Model Selection and Assessment]], and they interact, so search them **together** ([[Grid Search]]).

Related: [[SVM]], [[Kernel Methods]], [[Regression]], [[Loss Functions]].
