---
tags: [machine-learning, validation]
---

# Validation Schemas

The meta-algorithms. Simple concepts with a rich notation - the notation is what removes the ambiguity. Notation as in [[Notation]]: $\Lambda$ is the hyperparameter, $\Lambda_m$ one of its values, $H_{\Lambda_m}$ the corresponding hypothesis space.

---

## 1. Model selection: hold-out validation

- Split $D_l$ into $D_{TR}$ and $D_{VL}$, $tr+vl=l$.
- For each value $\Lambda_m$ **[grid search]**:
  - **[training]** $h^*_m(D_{TR})=\arg\min_{h\in H_{\Lambda_m}} R_{emp}(h,D_{TR})$
  - **[selection estimate]** $R_{emp}(h^*_m,D_{VL})=\frac{1}{vl}\sum_{z_i\in D_{VL}}L(z_i,h^*_m(D_{TR}))$
- Select $\Lambda^*_m=\arg\min_m R(h^*_m)$ **[best on VL]**
- **Return** $h^*(D_l)=\arg\min_{h\in H_{\Lambda^*_m}}R_{emp}(h,D_l)$ **[retraining on all $l$ data]**

> [!warning]
> It does **not** return the VL value as a test estimate. That number exists only for selection purposes.

---

## 2. Model selection: K-fold CV

- Split $D_l$ into $K$ distinct equal parts $D_1,\dots,D_K$.
- For each $\Lambda_m$:
  - For each part $D_k$ (with counterpart $\bar D_k$) **[folds cycle]**:
    - **[training]** $h^*_m(\bar D_k)=\arg\min_{h\in H_{\Lambda_m}}R_{emp}(h,\bar D_k)$
    - **[local estimate]** $R_{emp}(h^*_m(\bar D_k),D_k)=\frac{1}{|D_k|}\sum_{z_i\in D_k}L(z_i,h^*_m(\bar D_k))$
  - $R(h^*_m(D_l))\approx\frac1K\sum_k R(h^*_m(\bar D_k))$
- Select $\Lambda^*_m=\arg\min_m R(h^*_m(D_l))$ **[best over all the CVs]**
- **Return** $h^*(D_l)=\arg\min_{h\in H_{\Lambda^*_m}}R_{emp}(h,D_l)$ **[retraining on all $l$ data]**

> [!important] Note the ORDER
> Each iteration starts **from scratch**, with training and estimation local to the split. Otherwise, e.g. using a model trained on split 1 for split 2, you would be validating on data that was used in training.

---

## 3. Model assessment: hold-out

- Split $D_l$ into $D_{TR}$ and $D_{TS}$, $tr+ts=l$. **The test set is separated NOW**, not later.
- **[training]** $h^*(D_{TR})=\arg\min_{h\in H}R_{emp}(h,D_{TR})$ - this optimization can itself include a model selection on the $D_{TR}$ part.
- **[test error]** $R(h^*(D_{TR}))\approx R_{emp}(h^*(D_{TR}),D_{TS})=\frac{1}{ts}\sum_{z_i\in D_{TS}}L(z_i,h^*(D_{TR}))$

---

## 4. Model assessment: K-fold CV

- Split $D_l$ into $K$ parts.
- For each $D_k$: train on $\bar D_k$ (possibly including a model selection), estimate $R$ on $D_k$.
- $R(h^*(D_l))\approx\frac1K\sum_k R(h^*(\bar D_k))$ - the mean over the folds.
- $K=l$ $\to$ leave-one-out CV.

We do not violate the principle: the test-set results are never used for training or selection in any splitting/iteration.

---

## 5. Merging selection + assessment

When you want both the best model **and** its expected risk. Combinations:

- hold-out with TR / VL / TS: three separate sets;
- CV + test: cross-validate on the training set, then test on a separate set;
- external CV for test + internal hold-out for selection;
- **double / nested CV**: for each subset of a K-fold, a nested CV on the other $K-1$.

### 5a. TR - VL - TS (hold out)

- Split $D_l$ into $D_{TR},D_{VL},D_{TS}$.
- For each $\Lambda_m$: train on $D_{TR}$, compute $R_{emp}(h^*_m(D_{TR}),D_{VL})$.
- Select $\Lambda^*_m=\arg\min_m R_{emp}(h^*_m(D_{TR}),D_{VL})$ **[best on VL]**.
- **[retraining]** $h^*(D_{TR}\cup D_{VL})=\arg\min_{h\in H_{\Lambda^*_m}}R_{emp}(h,D_{TR}\cup D_{VL})$.
- Estimate $R$ with $\frac{1}{ts}\sum_{z_i\in D_{TS}}L(z_i,h^*(D_{TR}\cup D_{VL}))$.
- *Question: can we now retrain again on all $D_l$? Does that use the test results?*

### 5b. K-fold CV for selection + hold-out test

- Split $D_l$ into $D_{TR}$ (the "design set") and $D_{TS}$.
- For each $\Lambda_m$: estimate $R$ on $D_{TR}$ by a K-fold CV.
- Select the best $\Lambda^*_m$ from those CVs.
- **[retraining]** on the whole $D_{TR}$.
- Estimate $R$ on $D_{TS}$.

### 5c. K-fold CV for test + internal hold-out for selection

Each outer row does its own internal TR/VL split (arbitrary partitions, with shuffling).

> [!note]
> This gives an **estimate of the risk**, not a unique model: you potentially get a different model per row.

### 5d. Double / nested K-fold CV

- Split $D_l$ into $K$ parts.
- For each part $D_k$ **[external cycle]**:
  - select $h^*(\bar D_k)$ by an **internal** K'-fold CV on $\bar D_k$ (K' can differ from K);
  - **[test error for this external split]** $R_{emp}(h^*(\bar D_k),D_k)$.
- $R(h^*(D_l))\approx\frac1K\sum_k R(h^*(\bar D_k))$ - the mean over the **test** folds, plus its standard deviation.

> [!important]
> This does **not** provide a final model - you potentially have different hyperparameters for each external split. If you need one, run a **separate** model selection (hold-out or K-fold CV) afterwards, plus a possible final retraining.
> This does not violate the rules: the test error (for that class of models/algorithms) has already been estimated above, together with the variance, and the model will fall inside the estimated interval. Note the consequence: it is not true that the test always happens as the final step - here it comes *before* the last model selection phase, even if it feels counter-intuitive.
> Double CV exploits all the data better, for both selection and assessment, at the cost of a high computational effort.

---

## Other aspects

- Consider the different **trials** (e.g. different weight initializations for a NN) when computing the mean values for TR, VL, TS - see [[Multiple Minima]].
- Compare your results with other methods.
- Use statistical tests to verify significance.
- Verify your model on more than one data set.

Related: [[Model Selection and Assessment]], [[Cross Validation]], [[Validation Pitfalls]].
