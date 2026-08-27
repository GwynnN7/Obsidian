---
tags: [machine-learning, models, neural-networks]
---

# Attention and Transformers

A **transformer** is a deep learning architecture that relies on the parallel **multi-head attention** mechanism, instead of the recurrence of [[RNN]].

## Attention

An attention mechanism lets a model access **any** preceding point along the sequence, rather than only what survived in the state.

- The attention layer weights all the previous states according to a **learned measure of relevance**, so it can carry information about far-away tokens.
- Especially useful in language translation, where far-away context can be essential for the meaning of a word.
- First used to improve RNN models, then alone.
- All tokens are processed **simultaneously**, computing weights between them in successive layers - or better, between their **embeddings**: scaled dot-product units using query and word vectors to find the highest correlations among the words of a sentence. E.g. in *"see that girl"*, *"that"* is identified as correlated to *"girl"*.
- The architecture is completed by an encoder and/or decoder plus feed-forward networks, depending on the purpose. **Auto-regression** is used for generation.
- It applies to image *patches* too, instead of words $\to$ vision transformers.

**GPT** = Generative Pre-trained Transformer, used to generate text, speech, images, code.

## Why it is here

It is the alternative to a recurrent state for sequence processing: instead of *compressing* the past into a state ([[RNN]]) it *keeps* the past and learns where to look.

Related: [[RNN]], [[Deep Learning]], [[Neural Networks]].
