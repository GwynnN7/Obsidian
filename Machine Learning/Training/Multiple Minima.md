---
tags: [machine-learning, training]
---

# Multiple Minima

The loss of a NN is **not convex**: many local minima, plus saddle points. The result therefore depends on the starting weights.

## What to do

- **(!) Try a number of random starting configurations** - e.g. 5-10 or more training runs / trials.
- Take the **mean** of the errors and look at the **variance** to evaluate the model. A random single run will fall in that range, so the risk is under control unless the variance is high (and high variance is bad anyway - see [[Bias-Variance Tradeoff]]).
- If you need exactly one final model:
  - the **best** on the validation set - this is a model selection choice, so it needs a VL set;
  - the **median** - more cautious;
  - a **random** one - in that case you don't need a VL set (you can still avoid unlucky cases by looking at the training error);
  - or keep them all as a committee - see [[Ensemble Learning]].
- State which one you kept, whatever the criterion.

## Is a global minimum needed? No.

1. In ML we don't need a local **or** a global minimum of $R_{emp}$: we are looking for the minimum of $R$, which we cannot compute. We often stop at a point that is not even a stationary point of the training error.
2. A NN builds a **variable-size hypothesis space**: the VC-dim grows during training, so the training error goes to zero *while* the network becomes too complex. Stopping before this overtraining condition is needed anyway, independently of the local/global minima issue.

A local minimum with too high error is not really a big problem: you can see the final training error and restart if you are not happy.

Related: [[Weight Initialization]], [[Gradient Descent]], [[Early Stopping]], [[Validation Pitfalls]].

Same problem in the other subject: [[Local Search]] - hill climbing gets stuck in local maxima and uses random restarts for exactly this reason.
