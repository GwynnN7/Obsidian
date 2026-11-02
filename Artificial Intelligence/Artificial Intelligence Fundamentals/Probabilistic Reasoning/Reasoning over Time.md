---
tags: [artificial-intelligence-fundamental, theory]
---

# Reasoning over Time

## Continual learning and concept drift

Real environments are not static, which forces **continual / lifelong** learning paradigms. Changes in the underlying data distribution:

- **Virtual drift:** the problem space expands, but the previously learned concepts and rules stay valid.
- **Real drift:** the concepts themselves change, invalidating the previous model.

> [!note]
> This is what breaks the i.i.d. assumption behind [[Generalization and Risk]] - the whole risk framework assumes the test data come from the same distribution as the training data. Real drift means your carefully validated model is now measuring the wrong thing.

## Markov chains

A stochastic process described by a random variable $X$ evolving over discrete time $t$.

- **Time-homogeneous / stationary assumption:** the dynamics are governed by a fixed law - no "acceleration". Dropping it means the model depends on the specific time step.
- The current state $X_t$ depends on its past values; the process starts from a prior $P(X_0)$.

**Markov assumptions:**

- **First-order:** the future is conditionally independent of the whole past given the present,
  $$P(X_t\mid X_{0:t-1})=P(X_t\mid X_{t-1})$$
- **$n$-th order:** predicting $X_{t+1}$ needs the $n$ preceding states. A larger order improves the approximation but costs more.

Then: [[Hidden Markov Models]].

Related: [[Bayesian Networks]], [[RNN]] (the same "summarize the past in a state" idea, learned instead of specified).
