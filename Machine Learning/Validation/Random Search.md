---
tags: [machine-learning, validation]
---

# Random Search

Sample hyperparameter configurations randomly instead of enumerating a grid (Bergstra & Bengio, JMLR 2012).

Why it works better than it sounds:

- it avoids resampling the same values of the **influential** hyperparameters when some others turn out to be non-influential;
- the budget of samples is fixed **independently** of the number of hyperparameters.

Picture: with 2 hyperparameters and 100 evaluations, a $10\times10$ grid explores only **10 distinct values per axis**, while 100 random choices explore many more values per axis.

See [[Hyperparameter Search]], [[Grid Search]].
