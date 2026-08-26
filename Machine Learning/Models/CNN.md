---
tags: [machine-learning, models, neural-networks]
---

# CNN

Convolutional Neural Networks (LeNet / ConvNet). The point is **architecture design as a way to put prior information into a NN** - a not-fully-connected feed-forward NN built for a specific kind of problem.

## The motivating problem

Character recognition (ZIP code / MNIST): $16\times16$ grayscale images.

- A standard fully-connected NN with 256 inputs gets a 5%-20% misclassification rate, and many of the errors come from **lack of invariance** (rotations, shifts).
- Idea: the features of a character should be found **wherever** they appear in the image, so the same operation should be applied to different parts of the input.

Two architectural constraints follow:

1. **Local connections** - each unit sees only a small **receptive field**, so it extracts local features.
2. **Weight sharing** - the same weights are reused by units looking at different positions. This reduces the number of free parameters while still allowing complex connectivity.

## Convolution

The name comes from the convolution operator: a weighted average of a function $f$, weighted by another function $g$ that slides over it.

**1D case (a stream), 3-weight kernel:**

$$
out_t=\sum_{i=1}^{3}w_i\,x_{t+i-2}
$$

The weights are tuned by learning as usual $\to$ an **adaptive filter**; several units together act as a **filter bank**. This 1D sliding-window version is the **Time-Delay NN** (TDNN) - see [[RNN]], where the same idea appears as the "finite memory" approach.

**2D case (an image):** a kernel (e.g. $3\times3$) is moved over the image to produce an output **feature map**, i.e. the features extracted by that filter. In NN terms: we traverse the image with *the same* neuron, which uses local connections.

- **Stride:** how many pixels the kernel jumps. Stride 1 = one pixel shift; stride 2 = subsampling, which reduces the feature map.
- **Padding:** used to treat the border.
- Libraries usually implement it in the **cross-correlation** form.

## Pooling

Reduce the feature map by, over the neighbors of each output:

- subsampling (stride > 1), or
- a mean (average or weighted average), or
- a **max pool** (most common).

So instead of producing a value for each input pixel, we produce one value per rectangular set of pixels. E.g. max pooling with a $2\times2$ filter and stride 2.

Pooling also makes the representation approximately **invariant to small translations**: max/mean change little if a few inputs shift, so a high output stays high for a slightly translated version.

## The four ingredients

1. **Local connections** - detect local motifs/patterns.
2. **Shared weights** - translation invariance (invariant to the location of the pattern), *and* fewer free parameters while keeping the connections: this is **regularization by architecture**.
3. **Pooling** - reduce the dimension of the representation layer by layer (a pyramid), and help invariance to small shifts and distortions.
4. **Many layers** - 1 and 2 applied to all the feature maps, progressively abstracting features from the image.

Consequences:

- **Learnable filters:** what used to be hand-engineered in image processing is now learned. Independence from prior knowledge and human effort in feature design is the major advantage.
- **Stacking:** units in later feature maps cover larger and larger areas of the original image (a cone from the image to the final maps) - they are indirectly connected to a large part of it.
- The number of filters/units (i.e. feature maps) can be **increased** in higher layers, while the spatial size shrinks. Pooling keeps the number of feature maps the same.
- It makes possible the **composition of primitives**: low-level features, learned once, are reused in all the combinations of the successive layers $\to$ [[Deep Learning]].
- A historical instance of a deep neural network. Analogies with the animal visual cortex have been studied: a local receptive field produces the strongest response to a spatially local input pattern.

## Training

- Typically [[Backpropagation]], with the usual heuristics ([[Hyperparameter Guidelines]]) plus specializations for weight sharing and pooling.
- With huge networks and large amounts of data, many hyperparameters are fixed by experience or expert suggestion rather than by search - running [[Cross Validation]] over a large hyperparameter space would be far too expensive.

## Some history / numbers

- **LeCun's ZIP-code networks (1989):** five architectures, from no hidden layer up to Net-5 with 5194 connections but only **1060 weights** (weight sharing) and a 1.6% misclassification rate. Training error was zero for all of them on a reduced set (~300 images), so the test curve is what separates them.
- **MNIST** (60k train / 10k test) became the standard benchmark. Progress: 0.35% (2011), 0.23% with a committee of conv nets (2012), 0.21% (2016).
  > [!warning]
  > Too many uses of the same test data for these to still be significant, especially for improvements of 0.1% - exactly the leakage problem of [[Validation Pitfalls]].
- **Plain deep MLP (2010):** "good old on-line backprop" reached 0.35% on MNIST with a 6-layer 784-2500-2000-1500-1000-500-10 net, 12 million weights, 2 hours on GPU. What it needed: many layers, many neurons, **numerous deformed training images** (data augmentation as complexity control - see [[Regularization]]) and GPUs.
- **AlexNet (Krizhevsky, Sutskever, Hinton, NIPS 2012):** ImageNet, 1.3M images, 1000 classes; 60 million parameters, 500k neurons, five convolutional layers, some followed by max-pooling, two globally connected layers, a final 1000-way softmax; non-saturating neurons and an efficient GPU implementation; **dropout** to reduce overfitting in the globally connected layers. Top-5 error from ~26% to 18.9%.
- **DeepFace (2014):** 9 layers, convolutional; trained on 4M facial images of 4000+ identities; 97.25% on "same person?" vs 97.53% for humans.
- **Capsule networks (Sabour, Frosst, Hinton 2017):** a "capsule" is a subset of neurons in a layer that outputs both an instantiation parameter (is the entity there?) and pose parameters (position, orientation) relative to a canonical version. Uses a **routing** mechanism instead of max-pooling; better than a conv net at recognizing highly overlapping digits, without transforming image preprocessing.

## Tensors and GPUs

Matrix multiplication (dot product) is the typical linear operation done efficiently on GPU, parallelizing the sums and the independent products.

- In programming, a **tensor** is a multi-dimensional matrix with a linear operation called `tensordot`.
- Many kernel operations can be executed as one matrix multiplication - you could build a block matrix by hand, or simply express them as a `tensordot` between tensors (TensorFlow, NumPy).
- That is why tensor notation is used in ML: it parallelizes many linear operations on GPUs without hand-building complex block matrices.

## Related

- [[Neural Networks]] - the general MLP framework this specializes.
- [[Deep Learning]] - where the stacking story continues.
- [[Regularization]] - weight sharing and pooling as architectural regularization; dropout; data augmentation.
- [[RNN]] - the 1D sliding-window (TDNN) idea, and the other way of getting memory.
- [[Data and Preprocessing]] - structured data.
- [[Cellular Automata]] - the NCA uses the same fixed-kernel convolution and a shared update network.
