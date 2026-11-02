---
tags: [machine-learning, models]
---

# CNN

Convolutional Neural Networks. The point is **architecture design as a way to put prior information into a NN** - a not-fully-connected feed-forward NN built for a specific kind of problem.

## The result

A convolutional layer applies **one small kernel over every position** of its input. In 2D, with input $\boldsymbol{I}$ and kernel $\boldsymbol{K}$:

$$
S(i,j)=(\boldsymbol{I}*\boldsymbol{K})(i,j)=\sum_m\sum_n \boldsymbol{I}(i+m,\,j+n)\,\boldsymbol{K}(m,n)
$$

The output $S$ is a **feature map**: what that one filter found, everywhere. The size of the map, per spatial dimension:

$$
o=\left\lfloor\frac{i-k+2p}{s}\right\rfloor+1
$$

| knob | symbol | what it does |
|---|---|---|
| input size | $i$ | side of the input map |
| kernel / receptive field | $k$ | how much of the input one unit sees |
| padding | $p$ | pixels added at the border, to control the shrinking |
| stride | $s$ | how far the kernel jumps; $s>1$ subsamples |

> [!important] Two ingredients, everything else follows
> **Local connections** (a unit sees only its receptive field) and **weight sharing** (the same kernel at every position). From these two you get translation invariance, far fewer free parameters - regularization by architecture - and a receptive field that grows with depth.

> [!note]
> What is written above is technically **cross-correlation**: true convolution flips the kernel, $\boldsymbol{K}(m,n)\to\boldsymbol{K}(-m,-n)$. Since the kernel is learned, the flip is irrelevant and every library calls this convolution.

## The motivating problem

Character recognition on small grayscale images, e.g. $16\times16$.

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
- Analogies with the animal visual cortex: a local receptive field produces the strongest response to a spatially local input pattern.

## Training

- Typically [[Backpropagation]], with the usual heuristics ([[Hyperparameter Guidelines]]) plus specializations for weight sharing and pooling.
- With huge networks and large amounts of data, many hyperparameters are fixed by experience or expert suggestion rather than by search - running [[Cross Validation]] over a large hyperparameter space would be far too expensive.

## Related

- [[Neural Networks]] - the general MLP framework this specializes.
- [[Deep Learning]] - where the stacking story continues.
- [[Regularization]] - weight sharing and pooling as architectural regularization; dropout; data augmentation.
- [[RNN]] - the 1D sliding-window (TDNN) idea, and the other way of getting memory.
- [[Data and Preprocessing]] - structured data.
- [[Deep Graph Networks]] - the same two ingredients (local receptive field, weight sharing) moved onto a graph, where the neighbours are unordered.
- [[Representation Learning]] - learned filters instead of engineered features.
- [[Cellular Automata]] - the NCA uses the same fixed-kernel convolution and a shared update network.
