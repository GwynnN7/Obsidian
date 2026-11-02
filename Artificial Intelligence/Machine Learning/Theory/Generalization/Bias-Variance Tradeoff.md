---
tags: [machine-learning, theory]
---

# Bias-Variance Tradeoff

A training set is only **one** possible realization from the universe of data: different training sets give different models. Bias-variance decomposes the error expected over those realizations into three named pieces.

Setting: regression, squared error loss, target $y$.

## The result

$$
\text{expected prediction error at }\boldsymbol{x}=\text{variance}+\text{bias}^2+\text{noise}^2 .
$$

Written out, with $\bar h(\boldsymbol{x})=\mathbb{E}_P[h(\boldsymbol{x})]$ the model output at $\boldsymbol{x}$ **averaged over training sets**:

$$
\mathbb{E}_P\big[(y-h(\boldsymbol{x}))^2\big]=
\underbrace{\mathbb{E}_P\big[(h(\boldsymbol{x})-\bar h(\boldsymbol{x}))^2\big]}_{\text{variance}}
+\underbrace{\big(\bar h(\boldsymbol{x})-f(\boldsymbol{x})\big)^2}_{\text{bias}^2}
+\underbrace{\mathbb{E}_P\big[(y-f(\boldsymbol{x}))^2\big]}_{\text{noise}^2}
$$

$$
=\operatorname{Var}[h(\boldsymbol{x})]+\operatorname{Bias}[h(\boldsymbol{x})]^2+\sigma^2 .
$$

- **Bias:** $[\bar h(\boldsymbol{x})-f(\boldsymbol{x})]^2$, the systematic discrepancy between the true function and the model output averaged over training sets. High when $H$ is too small / too rigid.
- **Variance:** variability of $h(\boldsymbol{x})$ across different realizations of the training data. High when the model is too flexible.
- **Noise:** irreducible; the labels contain random error (e.g. more than one possible $d$ for the same $\boldsymbol{x}$). Even the optimal solution can be wrong. Does not depend on the model.

> [!important]
> Bias and variance are computed **across possible training sets**, not from one training run. There is a different $h$ - and a different $y$ - for every training set that could have been extracted.

## The two sources, informally

Fit a linear $h(\boldsymbol{x})=\boldsymbol{w}^T\boldsymbol{x}+w_0$ by least squares on $l$ examples, minimizing $\sum_{i=1}^{l}(y_i-h(\boldsymbol{x}_i))^2$:

1. because of the **hypothesis class we chose**, for some $f$ there is a systematic prediction error whatever the data $\to$ bias;
2. depending on **which data set we happen to get**, the $\boldsymbol{w}$ we find is different $\to$ variance.

The darts analogy: bias is the whole cluster sitting off-centre, variance is the cluster being spread out. It is a good picture, but the match with ML is not perfect - in ML we usually cannot see the individual throws.

## Derivation

### Setting

- Examples $\langle\boldsymbol{x},y\rangle$ where the true function is $y=f(\boldsymbol{x})+\varepsilon$, with $\varepsilon$ Gaussian noise of **zero mean** and standard deviation $\sigma$.
- Data points drawn i.i.d. from a single unknown distribution $P$.
- The quantity we want, for an arbitrary new point $\boldsymbol{x}$:

  $$
  \mathbb{E}_P\big[(y-h(\boldsymbol{x}))^2\big],
  $$

  where $y$ is a value for $\boldsymbol{x}$ that could be present in a data set, and the expectation is over **all training sets** drawn according to $P$.

### Variance lemma

Starting from $\bar Z=\mathbb{E}_P[Z]=\sum_i z_iP(z_i)$ and $\operatorname{Var}[Z]=\mathbb{E}[(Z-\bar Z)^2]$:

$$
\operatorname{Var}[Z]=\sum_i(z_i-\bar Z)^2P(z_i)=\sum_i(z_i^2-2z_i\bar Z+\bar Z^2)P(z_i)
$$

$$
=\sum_i z_i^2P(z_i)-2\bar Z\sum_i z_iP(z_i)+\bar Z^2\sum_i P(z_i)
=\mathbb{E}[Z^2]-2\bar Z\bar Z+\bar Z^2\cdot 1=\mathbb{E}[Z^2]-\bar Z^2 .
$$

It gets used in the form

$$
\mathbb{E}[Z^2]=\bar Z^2+\operatorname{Var}[Z],
$$

applied twice: once to $h$ over the training sets, once to the target with its noise.

### Step 1 - expand the square

$$
\mathbb{E}_P\big[(y-h(\boldsymbol{x}))^2\big]
=\mathbb{E}_P\big[h(\boldsymbol{x})^2-2yh(\boldsymbol{x})+y^2\big]
=\mathbb{E}_P\big[h(\boldsymbol{x})^2\big]+\mathbb{E}_P\big[y^2\big]-2\,\mathbb{E}_P[y]\,\mathbb{E}_P[h(\boldsymbol{x})].
$$

The cross term splits because the noise on the new point and the training set that produced $h$ are independent.

### Step 2 - first term, by the lemma

Let $\bar h(\boldsymbol{x})=\mathbb{E}_P[h(\boldsymbol{x})]$ be the **mean prediction** at $\boldsymbol{x}$, when $h$ is trained with data drawn from $P$. Then

$$
\mathbb{E}_P\big[h(\boldsymbol{x})^2\big]=\mathbb{E}_P\big[(h(\boldsymbol{x})-\bar h(\boldsymbol{x}))^2\big]+\bar h(\boldsymbol{x})^2 .
$$

### Step 3 - second term, by the lemma

Zero-mean noise gives $\mathbb{E}_P[y]=\mathbb{E}_P[f(\boldsymbol{x})+\varepsilon]=f(\boldsymbol{x})$, so $f(\boldsymbol{x})$ plays the role of $\bar Z$ and

$$
\mathbb{E}_P\big[y^2\big]=\mathbb{E}_P\big[(y-f(\boldsymbol{x}))^2\big]+f(\boldsymbol{x})^2 .
$$

### Step 4 - put everything back

Substituting steps 2 and 3 into step 1, and using $\mathbb{E}_P[y]=f(\boldsymbol{x})$, $\mathbb{E}_P[h(\boldsymbol{x})]=\bar h(\boldsymbol{x})$ in the cross term:

$$
\mathbb{E}_P\big[(y-h(\boldsymbol{x}))^2\big]=
\mathbb{E}_P\big[(h(\boldsymbol{x})-\bar h(\boldsymbol{x}))^2\big]
+\bar h(\boldsymbol{x})^2-2f(\boldsymbol{x})\bar h(\boldsymbol{x})+f(\boldsymbol{x})^2
+\mathbb{E}_P\big[(y-f(\boldsymbol{x}))^2\big].
$$

The three middle terms are a square, $\bar h^2-2f\bar h+f^2=(\bar h-f)^2$, so

$$
\begin{aligned}
\mathbb{E}_P\big[(y-h(\boldsymbol{x}))^2\big]
&=\mathbb{E}_P\big[(h(\boldsymbol{x})-\bar h(\boldsymbol{x}))^2\big] &&\text{variance}\\
&\quad+\big(\bar h(\boldsymbol{x})-f(\boldsymbol{x})\big)^2 &&\text{bias}^2\\
&\quad+\mathbb{E}_P\big[(y-f(\boldsymbol{x}))^2\big] &&\text{noise}^2
\end{aligned}
$$

and since $y-f(\boldsymbol{x})=\varepsilon$, the last term is $\mathbb{E}_P[\varepsilon^2]=\sigma^2$. Hence

$$
\mathbb{E}_P\big[(y-h(\boldsymbol{x}))^2\big]=\operatorname{Var}[h(\boldsymbol{x})]+\operatorname{Bias}[h(\boldsymbol{x})]^2+\sigma^2 .
$$

## Capacity and regularization

Same experiment (25 data sets from a sinusoid, RBF network with LMS), varying $\lambda$ in [[Regularization]] - recall $\operatorname{Loss}(\boldsymbol{w})=\sum_p(d_p-o(\boldsymbol{x}_p))^2+\lambda\lVert \boldsymbol{w}\rVert^2$, data term plus penalty term:

| $\lambda$ | effect |
|---|---|
| high | penalty dominates, model too rigid: **high bias, low variance** $\to$ [[Underfitting]] |
| medium | the trade-off |
| low | strongly dependent on the specific training data: **low bias, high variance** $\to$ [[Overfitting]] |

With very low $\lambda$ the single fits are wild, but the **mean** $\bar h(\boldsymbol{x})$ is still close to the true function - which is exactly the idea behind [[Bagging]].

> [!warning]
> On a plot indexed by $\lambda$, complexity grows to the **left** - the reverse of the usual complexity plots. Read the axis before reading the curves.

## Reading the error-vs-complexity plot

Plot training and test error against model complexity for many training sets of the same size:

- the training curves fall monotonically and stay bunched together;
- the test curves are U-shaped and **fan out** as complexity grows - the spread between a lucky and an unlucky training set *is* the variance;
- the minimum of the average test curve is the trade-off point.

Complex model $\to$ lower bias, higher variance $\to$ possible [[Overfitting]].

> [!note]
> This U-shape is the classical picture and it is about **capacity**, not about the parameter count. Push the parameters far past the point of zero training error and the test error can start falling again - **double descent**, and its epoch-wise cousin **grokking**, both in [[Deep Learning]].

## Why it matters for validation

The decomposition explains why estimating a model's performance is hard: a single train/test split can be lucky or unlucky. Hence repeated splits, [[Cross Validation]], and reporting mean $\pm$ std.

## Ensembles

- [[Ensemble Learning]] - combine several models.
- [[Bagging]] - mainly reduces **variance**: averaging over resampled training sets is a direct attack on the variance term.
- [[Boosting]] - mainly reduces **bias**.

## Related

- [[Statistical Learning Theory]] - same trade-off, analytical version.
- [[Regularization]], [[Model Selection and Assessment]], [[Feature Selection]].
