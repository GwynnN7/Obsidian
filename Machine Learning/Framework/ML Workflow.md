---
tags: [machine-learning, framework, practical]
---

# ML Workflow

The design cycle, in order:

1. **Collect data** - selection, integration, cleaning. Needs an adequately large and representative set for both training and test.
2. **Represent data** - domain dependent, uses prior knowledge of the application expert: encoding, variable scaling, [[Feature Selection]], outlier detection, missing data. *Often the most critical phase for the overall success.*
3. **Choose the model** - state the problem, formulate the hypothesis, know the limits of applicability of the model, plan the complexity control.
4. **Train** - the learning algorithm on the training data.
5. **Evaluate** - predictive accuracy first, then interpretation of the outcomes. See [[Validation]].
6. **Deploy**.

Prior knowledge feeds into steps 2 and 3.

## Warnings

- **Causality is assumed, not discovered.** Data analysis alone cannot infer it - the formal treatment is the do-operator of [[Causal Networks]]. People in Florida are older on average than in other US states; that does not mean the Florida climate makes people live longer.
- Uninformative input variables $\to$ poor modelling $\to$ poor results.
- Powerful models fit "garbage" data too, so a flexible model raises the risk rather than lowering it.
- Badly validated results are misleading both in the prediction and in the interpretation.

## Related

- [[Hyperparameter Guidelines]] - what to set and how, in practice.
- [[Validation Pitfalls]] - the mistakes that actually happen.
