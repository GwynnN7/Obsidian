---
tags: [machine-learning, models, neural-networks]
---

# Representation Learning

> [!important] The idea
> Feed the machine with **raw data** and let it automatically discover the representation needed for the task, instead of engineering the features by hand.

Many tasks are easy or hard depending only on **how the information is represented**. A good representation is one that makes the subsequent learning task easier.

- Manual design of features takes decades of work in specific communities (language, images, chemistry). In the vault's terms, it means hand-picking the $\phi(\boldsymbol{x})$ of [[Linear Basis Expansion]].
- A supervised MLP already does it automatically: every hidden layer takes on properties that make the output layer's task easier, because $\phi(\boldsymbol{x},\boldsymbol{w})$ is **learned** by fitting $\boldsymbol{w}$ - see [[Neural Networks]] and the XOR example.
- [[CNN]] does it on raw images: no pattern or engineered feature is specified in advance.
- [[Deep Learning]] is representation learning with **multiple levels** of representation, each expressed in terms of simpler ones.

The concept is more general than deep models, though: it applies to NN in general and to probabilistic models with latent variables. Note it normally refers to raw data (images, text) - not to the case where you already provide a small set of good features.

## Two questions

| | |
|---|---|
| How to **obtain** a hidden representation? | semi-supervised learning $\to$ **pre-training** |
| How to **exploit** one you already have? | **transfer learning** |

## Autoencoders

An autoencoder is a NN trained to **copy its input to its output**. The hidden layer $\boldsymbol{h}$ is a **code** for the input, and the network splits in two:

$$
\boldsymbol{h}=f(\boldsymbol{x}) \quad \text{(encoder)},\qquad \boldsymbol{r}=g(\boldsymbol{h}) \quad \text{(decoder, reconstruction)}.
$$

Training is **unsupervised**: the target is the input itself.

- **Undercomplete** - the hidden layer is *smaller* than the input, so the architecture itself forces the code to keep only the most salient features. With a linear decoder and an LMS criterion you recover PCA.
- **Overcomplete** - the hidden layer is *larger* than the input. Now copying is trivial, so you need [[Regularization|regularization]] to impose sparsity of the representation, robustness to noise, or whatever property you want.

**Denoising autoencoders / auto-associative memories:** train on a set of images, present a noisy version and reconstruct the clean one by repeated encoding/decoding. Here a good representation is *one that can be obtained robustly from a corrupted input and is useful for recovering the clean one*.

Variants worth knowing:

- **Variational autoencoders** - the latent space is a mixture of distributions instead of a fixed vector, learned in a Bayesian framework.
- **Encoder-decoder for translation** - the output does not have to match the input: sequences in, sequences in another language out.
- **Transformers** are encoder/decoder networks plus attention; the encoder alone is used for representation, the decoder alone for generation $\to$ [[Attention and Transformers]].

## Greedy layer-wise pre-training

The first approach that made training a deep supervised network possible (outside [[CNN]] and [[RNN]]): use **unsupervised** learning, layer by layer, to capture the shape of the input distribution, then fine-tune.

1. Train the first layer as an auto-associator minimizing the reconstruction error of the raw input. Unsupervised w.r.t. the global task, so **unlabeled** examples are enough.
2. Use the hidden outputs of that auto-associator as the input of another layer, also trained as an auto-associator.
3. Iterate to add the desired number of layers.
4. Take the last hidden layer output as input to a **supervised** layer, and initialize its parameters (randomly, or by training it with the rest frozen).
5. **Fine-tune** all the parameters of the deep architecture with respect to the supervised criterion.

The hope: the greedy unsupervised initialization puts all the layers in a region of parameter space from which a good local minimum is reachable by local descent.

Implemented either with stacked (denoising) autoencoders, or with Restricted Boltzmann Machines - energy-based models, stacked into deep belief networks.

### Why it helps

- **Initialization:** a good start is half the work, it drives the weights where a better local minimum is reachable $\to$ [[Multiple Minima]].
- **Regularization:** not by making the model simpler, but by discovering features shaped by the regularities of the input distribution. If the true function is complicated but shaped by those regularities, unsupervised learning is a more appropriate regularizer than a norm penalty.
- **Variance reduction** of the training process: pre-trained models tend to land in a smaller region.

> [!note] Still needed?
> Largely abandoned for general use: the techniques in [[Deep Learning]] are usually enough to train from scratch end-to-end, and pre-training is awkward to manage (hyperparameters split over two phases). It still pays off in NLP, where huge unlabeled corpora give distributed representations of words - see [[Distributed Representation]].

## Transfer learning

Use the representation discovered by one model to improve another. The assumption: **there are features useful for more than one setting or task**, corresponding to underlying factors that appear in both.

- Going from the autoencoder to the classifier above is already a form of it.
- **Multi-task learning:** same inputs, different targets, one shared representation $\to$ an implicit [[Regularization|regularization]].
- **Domain adaptation:** different input domain, shared features (e.g. sentiment analysis moved from one kind of product to another).
- **Pre-trained backbones:** take a CNN trained on a large image collection, keep the low layers (the primitives are shared among all images), replace the last layers with the ones for your task, and train only those - or fine-tune the whole thing - with a much smaller data set.

> [!note]
> These are sub-fields of ML, not just terminology.

## Related

[[Deep Learning]], [[Distributed Representation]], [[Neural Networks]], [[CNN]], [[Semi-Supervised Learning]], [[Unsupervised Learning]], [[Linear Basis Expansion]].
