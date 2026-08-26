---
tags: [machine-learning, models, neural-networks]
---

# Distributed Representation

> [!important]
> In a distributed representation the elements (the features) are **not mutually exclusive**, and their many configurations correspond to the variations seen in the data.

The opposite is the **symbolic / localist / one-hot** representation, where one unit is 1 and all the others are 0.

| | Symbolic (one-hot) | Distributed |
|---|---|---|
| Values | $0/1$, exactly one active | real numbers (dense), a whole activation pattern |
| Distance between two concepts | always $\sqrt2$ | reflects the meaning |
| Sharing | none | concepts share features |
| Generalization | one concept at a time | shared attributes transfer between concepts |

> [!warning] We are representing, not classifying
> A vector of activations here is a **code** for a concept, not a set of class scores.

## Input or hidden layers?

The concept is general, but learning acts on the **internal** representation - that is the representation-learning part, see [[Representation Learning]].

- On the **input** you would need background knowledge to fix a distributed encoding, which is difficult and rare; fixing it arbitrarily can hamper the task.
- So typically the input is one-hot (uninformed, see [[Data and Preprocessing]]) and the model develops the distributed representation internally.
- Models differ in whether they can exploit distributed representations internally at all.

## Counting the difference

Blue car, blue bike, red car, red bike: 4 concepts, but you do not need 4 neurons. **2 are enough**: one for blue/red, one for car/bike.

In general, $n$ features with $k$ values each describe $k^n$ concepts - in binary, $2^n$ configurations against the $n$ concepts a one-hot code would give.

So a model based on distributed representations (a NN with hidden units) can distinguish **exponentially more regions** of the input space than a non-distributed one - which is the same non-local-generalization argument as in [[Curse of Dimensionality]] and [[Deep Learning]].

Models that rely on **non**-distributed representations:

- [[k-NN]] - one or few prototypes per input, no sharing across the training data;
- RBF kernels - reduce to k-NN, see [[Kernel Methods]];
- mixtures of gaussians - the same;
- decision trees - only one leaf (and the path from the root) is active.

## Disentangling concepts

The same example, now for the meaning:

| | car (vs bike) | red (vs blue) |
|---|---|---|
| red car | 1 | 1 |
| red bike | 0 | 1 |
| blue car | 1 | 0 |
| blue bike | 0 | 0 |

Sharing a **column** disentangles the concept of "redness", shared among different objects. So the model can learn about the colour without having to see every combination of the other features.

- The neuron describing redness learns about redness from images of cars **and** of bikes.
- **Statistical separability** $\to$ it generalizes to configurations never seen during training: having seen the other three, blue bike is not a completely new thing.
- Same story for combining learned factors: gender, glasses, and so on, recombined to represent a case never presented.

## Word embedding

The clearest example. Instead of a one-hot code per word (dimension = vocabulary size, every word at distance $\sqrt2$ from every other), map words into a vector space:

1. **Treat similar words similarly.** If *dog* and *cat* share many attributes, sentences containing *cat* inform the predictions made for sentences containing *dog*, and vice versa - information transfers from one training sentence to an exponentially large number of semantically related ones.
2. **Learn that similarity automatically**, from the contexts in which the words appear (word2vec-style techniques).

The result: words with similar meaning end up as neighbours, and whole regions of the space become semantically coherent (countries in one zone, years in another). The representations are obtained by backpropagation, jointly learning a vector per word and a function that predicts a target - the next word for language modelling, or a translated sequence.

Arithmetic in that space carries meaning:

$$
\text{Rep}(king)-\text{Rep}(male)+\text{Rep}(female)\approx\text{Rep}(queen)
$$

$$
\text{Rep}(Paris)-\text{Rep}(France)+\text{Rep}(Poland)\approx\text{Rep}(Warsaw)
$$

i.e. the difference vector captures the relationship (gender, capital city).

> [!note]
> The model is not only learning the grammar - the learned representations are semantically significant, and nothing but text was provided. Why *our* semantics corresponds to what the network learns is an open question; the working hypothesis is that language and images have a hierarchical structure of their own, and a deep NN has the right inductive bias for it (real images are a tiny fraction of all the possible white-noise images).

## The price

**Interpretability.** A one-hot code says exactly what the unit means; a distributed code spreads the meaning over many units, and reading it back is much harder.

## And with depth

[[Deep Learning]] exploits distributed representations **through many layers**, composing levels of abstraction and reusing features. That gives **two** stacked exponential advantages: one over non-distributed representations, one over shallow models.

Globally: a deep model learns a distributed representation of the data, disentangling the shared causal factors that generated them, across different levels of abstraction.

## Related

[[Representation Learning]], [[Deep Learning]], [[Neural Networks]], [[Attention and Transformers]], [[RNN]], [[Data and Preprocessing]].
