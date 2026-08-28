---
tags: [machine-learning, validation]
---

# Validation Pitfalls

The FAQ / typical-errors page. Two mementos to keep in mind:

1. Estimating the error on TR and/or VL is **not** a good risk estimation.
2. Using the entire data set for feature/model selection **prejudices** the estimation.

---

## The counterexample (why assessment must be correct)

- 20-30 examples, 1000 random input variables, random $0/1$ target.
- We select 1 model using 1 input feature that guesses (by chance) 99% or 100% on any successive TR/VL/TS splitting.
- Perfect result? **No.** The right test error is 50% - a coin.

What went wrong:

- error on TR and VL is not a risk estimation;
- the entire data set was used for feature/model selection $\to$ biased estimation, called **subset selection bias**;
- the test set was implicitly used at the very beginning.

> [!warning]
> The test set must be separated **in advance**, before making any model selection - even before feature selection. On a truly external test the result is 50%.
> And K-fold CV does **not** solve this by itself (it can even confuse you).

We are discussing the **correctness of the estimation**, not the possibility of solving the task.

## Overfitting with CV

Yes, intensive use of CV can overfit, and it keeps the same problem. Example on feature selection: 20-30 records, 1000 attributes, try 1000 linear regressions each using one attribute; the best of the 1000 looks great on the validation set - but it would have looked great with a purely random output too.

**Why:** the selection was made on the basis of all the examples.
**Fix:** leave samples out *before* selecting.

- Hold out an additional test set before doing **any** model selection, and check the best model there too.
- Model selection with CV: use "CV + test" or double CV (a new selection **for each fold**).
- Assessment with CV: do the feature selection **inside each fold** $\to$ you correctly get $\approx50\%$ on the random task.

---

## FAQ: stopping - where?

Avoid stopping NN training at an arbitrary fixed number of epochs. Details in [[Stopping Criteria]] and [[Early Stopping]].

## FAQ: early stopping inside CV

- ES criteria are part of model selection - not on the TS, and in principle they need a VL set **any time you train the model**.
- Problem: after a CV you have a different stop point per fold. How do you pick one for the final retraining?
  - Treat #epochs as a hyperparameter and take the mean over the folds. Issue: changing the data size for retraining can move the best stop point.
  - **Much better:** take the mean of the **TR error** at the best VL point and use it to reach the same level of fitting in the retraining (avoids undertraining).
  - Or keep a VL set during the retraining too.
- Your choice: ES, or rely on a regularized model with standard stopping criteria. Regularization anyway avoids strong overtraining and the strict need of ES.

## FAQ: random initialization

Different $W_{initial}$ $\to$ different models. How to pick one for the final model (which you should always state)?

- Compute mean and variance of the error/accuracy over the trials: a random case will fall in that range, so the risk is under control **unless** the variance is high (and high variance is bad in general).
- Take all of them - see [[Ensemble Learning]].
- If you pick one, it is a **model selection** $\to$ you need a VL set: the best (min VL error), or the median (more cautious). Picking at random does not need a VL set (you can still avoid unlucky cases by looking at the training error).

See [[Multiple Minima]].

## Typical error: sequential selection of hyperparameters

The order introduces a bias, e.g. an $\eta$ tuned on 10 units then reused for 100 units.

- After initial trials to find the right range/interval (looking at the stability of training) and the most influential hyperparameters (looking at the variance of the results), search on a **grid with all the combinations**.
- The grid search can be repeated to refine a smaller range found before (nested grid searches, coarse to fine), but all the significant hyperparameters must stay **together**, to account for the cross effects.
- Think and select the relevant aspects: don't just throw an automatic expensive search at it. Time constraints push you towards a basic, fast, still correct schema.

## The final reflection

*What do I do after model selection and assessment? Can I make a new design on that basis?*

- No, and it is not really advantageous either (unless you get new data for a new test set).
  - Repeating this with the same data you can reach **any** result you like on that data set - your model can be perfect on it and useless later; you are no longer estimating the risk.
  - We shouldn't tune the predictor on the test error, because we wouldn't be able to do that on future examples.
  - A blind test in a competition does not even allow it.
  - But your internal test set is useful to **check your approach**.
- People are tempted to decide on the basis of the results; that is a psychological problem outside ML methods. At least be aware you are doing it. Better:
  - want a good model $\to$ do a good model selection;
  - want to know the real performance $\to$ do a rigorous test estimation.

## In practice

- Simple hold-out is widely applied; sometimes the test set is already given separately.
- 5- or 10-fold CV is the recommended compromise vs LOO.
- Check your TR and VL error rates. See the warnings for few data in [[Cross Validation]].

Related: [[Model Selection and Assessment]], [[Validation Schemas]], [[Feature Selection]].
