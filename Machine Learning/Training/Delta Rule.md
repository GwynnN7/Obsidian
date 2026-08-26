---
tags: [machine-learning, training]
---

# Delta Rule

Error-correction update (Widrow-Hoff / Adaline / LMS rule) for a differentiable output:

$$
w_{new}=w+\eta\,\delta\,x,\qquad \delta=(d-out).
$$

Why it is an "error correction" rule:

- $\delta=0$ $\to$ no correction;
- input $>0$ and error positive (output too low) $\to$ increase $w_j$ $\to$ increase the output $\to$ less error;
- input $>0$ and error negative (output too high) $\to$ decrease $w_j$.

In neuron terms: the adjustment of a synaptic weight is proportional to the product of the error signal and the input signal that excites the synapse. Easy to compute **when the desired response of the unit is known** - which is not the case for hidden units, hence [[Backpropagation]].

With a sigmoidal unit the delta becomes

$$
\delta_p=(d_p-o(x_p))\,f'_\sigma(net(x_p)).
$$

Compare with the [[Perceptron]] rule, where $out=\operatorname{sign}(w^Tx)$: the Hebbian form $w_{new}=w+\eta\,d\,x$ vs the error-correction form above.

Related: [[Linear Models]], [[Neural Networks]], [[Gradient Descent]].
