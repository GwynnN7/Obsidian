# Univariate Optimization

For $\min_{x\in I}f(x)$, distinguish **global** methods, which search an interval under strong assumptions, from **local** methods, which converge near one minimum.

## Existence and localization

A continuous function on a compact interval $[a,b]$ attains a minimum. On an open or unbounded set, an infimum may not be attained. A local minimum $x^*$ in the interior satisfies $f'(x^*)=0$ if $f$ is differentiable, but stationary points can also be maxima or inflections.

## Unimodality

$f$ is unimodal on $[a,b]$ when it decreases up to a minimizer and then increases. Comparing values at two interior points then lets us discard a whole subinterval. Without unimodality, interval reduction may discard the global minimum.

## Three information levels

- Function values only: bracketing, dichotomic/golden-section search, interpolation.
- First derivatives: bisection on $f'$ when derivative signs bracket a stationary point.
- First and second derivatives: Newton's method.

## Bracketing a local minimum

A triple $a<b<c$ brackets a minimum if $f(b)<f(a)$ and $f(b)<f(c)$. A practical exploratory phase increases step lengths until this pattern appears. Bracketing controls where a local method searches and provides a fallback if interpolation/Newton steps misbehave.

## Global Lipschitz idea

If $|f(x)-f(y)|\le L|x-y|$, each sampled value gives a lower cone $f(x_i)-L|x-x_i|$. The lower envelope is a global lower bound; evaluate where it is smallest and refine. This can certify global accuracy but becomes expensive, and knowing a useful $L$ is difficult.

## Stopping

Use a bracket-width test $b-a\le\varepsilon_x(1+|x|)$ and/or a function/derivative test. A tiny derivative is not by itself proof of a minimum; verify local curvature or surrounding values.

Next: [[Bracketing and Dichotomic Search]] and [[Interpolation Newton and Convexity]].


