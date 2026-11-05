# Optimization Problems

An optimization problem is
$$
(P)\qquad f^*=\inf\{f(x):x\in X\}.
$$
$X\subseteq\mathbb R^n$ is the feasible region. A feasible $x^*$ is an optimal solution when $f(x^*)\le f(x)$ for every $x\in X$. The optimal value may exist even when no solution attains it, e.g. $\inf\{x:x>0\}=0$.

## Local, global, and approximate solutions

- **Global minimum:** best over all $X$.
- **Local minimum:** best within some neighborhood of $x^*$.
- **Strict local minimum:** every other nearby feasible point is worse.
- **$\varepsilon$-optimal:** $f(x)\le f^*+\varepsilon$ (when $f^*$ is finite).

Floating-point computation makes exact comparisons unrealistic. A practical test normally combines objective progress, step size, gradient norm, and a maximum-iteration safeguard. Relative tolerances are often better than absolute ones because scale matters.

## Reformulations

Reformulations should preserve solutions while producing a more useful representation.

- Maximizing $f$ is minimizing $-f$.
- A constraint $g(x)\ge0$ becomes $-g(x)\le0$.
- $g(x)=\delta$ becomes $g(x)-\delta=0$.
- $g(x)\le0$ can become $g(x)+s=0$, $s\ge0$ using a slack variable.
- Several inequalities are equivalent to $\max_i g_i(x)\le0$, though this creates a nonsmooth max.
- Bounds $\ell\le x\le u$ form a box and are especially easy to handle.

## Domains and extended values

If $f$ is undefined outside a set $D$, either keep $D$ explicit or extend $f$ with $+\infty$. The indicator
$$
\iota_X(x)=\begin{cases}0&x\in X,\\+\infty&x\notin X\end{cases}
$$
turns $\min_{x\in X}f(x)$ into $\min_x f(x)+\iota_X(x)$. This is mathematically neat but can hide exploitable constraint structure.

## Multiple objectives

With several criteria, vectors are only partially ordered. A point is **Pareto optimal** if no feasible point improves one objective without worsening another. A weighted sum $\sum_iw_if_i(x)$ converts criteria to one objective, but the weights encode preferences and may miss nonconvex parts of the Pareto frontier.

## Practical checklist

1. What are the variables, objective, and feasible set?
2. Is the problem bounded below and is the infimum attained?
3. Are function/constraint evaluations numerically meaningful at the intended scale?
4. Is there useful structure: linearity, separability, quadratic form, smoothness, convexity?
5. What does an acceptable approximate solution mean?

See also [[Elementary Optimization Models]] and [[Convex Functions|Convex Functions]].


