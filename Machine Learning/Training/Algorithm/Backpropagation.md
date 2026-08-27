---
tags: [machine-learning, training, neural-networks]
---

# Backpropagation

Training algorithm for a feed-forward MLP: compute the gradient of the total error efficiently by propagating error signals **backward** through the network.

The problem it solves: estimate the contribution of the **hidden** units to the error at the output level (the credit assignment problem). The answer is the *Generalized Delta Rule*.

## Training loop

1. Forward pass: compute unit outputs.
2. Compute output errors and output-layer delta values.
3. Propagate delta values backwards to the hidden layers.
4. Update all weights, **including bias weights**.
5. Repeat until a valid stopping criterion; see [[Stopping Criteria]] and [[Early Stopping]].

# Training Algorithm
### Feedforward Step
Forward pass to calculate the output of each hidden layer and forward it to the next layer
$$
net_t=\sum_s w_{ts}o_s, \quad o_t = f_t(net_t)
$$

Upon reaching the output layer, for one pattern $p$ and $K$ output units:

$$
E_p=\frac12\sum_{k=1}^{K}(d_k-o_k)^2,
\qquad E_{tot}=\sum_pE_p.
$$

Goal: minimize the error $E_{tot}$ (or a regularized loss) with [[Gradient Descent]], given a Training Set defined as:
$$
D_{TR}=\{(x^1,d^1),\dots,(x^l,d^l)\}.
$$

To simplify, $p$ is omitted below: $o_k$ stands for $o_k(x_p)$ and $d_k$ for $d_{p,k}$.

### Backpropagation Step
The purpose is to find the derivative of $E_{tot}$ with respect to the weights to minimize the loss
$$
	\Delta w=-\frac{\partial E_{tot}}{\partial w} = -\sum_p\frac{\partial E_p}{\partial w} = \sum_p \Delta_p w
$$
For the **Chain Rule of Calculus**, given that each gradient to reach a generic $w_{tu}$ from $E_p$ has to pass through $o_t$ and through $net_t$, we have:
$$
-\frac{\partial E_p}{\partial w_{tu}} = -\frac{\partial E_p}{\partial o_{t}}\frac{\partial o_t}{\partial net_t}\frac{\partial net_t}{\partial w_{tu}} = -\frac{\partial E_p}{\partial o_{t}}f'_t(net_t)o_u = \delta_t o_u
$$
Going step by step:
$$
\Delta_p w_{tu} = -\frac{\partial E_p}{\partial w_{tu}} = -\frac{\partial E_p}{\partial net_{t}}\frac{\partial net_t}{\partial w_{tu}} = \delta_t o_u
$$
$$
-\frac{\partial E_p}{\partial net_{t}} = \delta_t, \quad  \frac{\partial net_t}{\partial w_{tu}} = \frac{\partial \sum_s w_{ts}o_s}{\partial w_{tu}} = o_u \quad \text{not 0 only for $s=u$}
$$
$$
\delta_t = -\frac{\partial E_p}{\partial net_{t}} = -\frac{\partial E_p}{\partial o_{t}}\frac{\partial o_t}{\partial net_{t}} = -\frac{\partial E_p}{\partial o_{t}} f'_t(net_t)
$$
- Case Output Unit ($t = k$)
 $$
 -\frac{\partial E_p}{\partial o_k} = -\frac{\partial \frac{1}{2}\sum_r^K(d_r - o_r)^2}{\partial o_k} = (d_k-o_k)
$$
$$
\delta_k = -\frac{\partial E_p}{\partial o_k}f_k'(net_k) = (d_k-o_k)f_k'(net_k)  
$$
- Case Hidden Unit ($t=j$)
$$
-\frac{\partial E_p}{\partial o_j} = \sum_k -\frac{\partial E_p}{\partial net_{k}} \frac{\partial net_k}{\partial o_{j}} = \sum_k \delta_k  \frac{\partial \sum_s w_{ks}o_s}{\partial o_j} = \sum_k \delta_k w_{kj}
$$

$$
\delta_j = -\frac{\partial E_p}{\partial o_j}f_j'(net_j) = (\sum_k \delta_k w_{kj}) f_j'(net_j)  
$$

Notes on the hidden case:
- it expresses the variation of $E_p$ considering **all** the output units $o_k$, hence the sum over $k$: each $o_k$ (and $net_k$) depends on $o_j$;
- crucially, $-\partial E_p/\partial net_k=\delta_k$ has **already been computed** at the output layer $\to$ we can back-propagate the $\delta_k$ to the hidden unit $j$. Hence the name.

### Conclusion
$$
\Delta_p w_{tu} = \delta_to_u, \quad \delta_k = (d_k-o_k)f_k'(net_k), \quad \delta_j =  (\sum_k \delta_k w_{kj}) f_j'(net_j)  
$$
$$
w_{tu}^{new} = w_{tu} + \eta\delta_to_u
$$

In particular $\Delta_p w_{kj}=\delta_k o_j$ (output unit, input from hidden $j$) and $\Delta_p w_{ji}=\delta_j o_i$ (hidden unit, input $i$).

This is the **on-line** version, using $\Delta_p w$ per pattern; for the batch version sum over the patterns. See [[Batch Online and Mini-Batch]].

## Where else it applies

Unfold a recurrent network through time and you get a deep feedforward network with shared weights, on which the very same algorithm runs - that is **BPTT**, see [[RNN]]. The same holds for the shared kernels of a [[CNN]].

## Generalization to $m$ hidden layers

The same derivation applies: the deltas come not only from the output layer but from any generic layer above, i.e. from **any unit the current unit is connected to**.

## Properties

> [!note]
> Backpropagation is a **training algorithm**, not a type of neural network. Don't say "backprop NN".

- **Compositional:** easy to program, because it follows the structure of the model.
- **Local:** it keeps track only of quantities local to each unit. We use only information around the unit (below or above), the rest arrives indirectly through the flow of deltas. Despite the local computation, the weight changes have a **global** effect on the network - the complex effect of $w_{tu}$ on the total error is decomposed into the local $\delta$ values.
- **Efficiency:** $O(\#W)$ instead of $O(\#W^2)$, thanks to the "magic factorization": the same $\delta_t$, computed once, is reused for all the weights of unit $t$, instead of recomputing something for each couple $(t,u)$. With $10^9$ weights this is $10^9$ vs $10^{18}$ operations.
- Requires: differentiable loss, differentiable activation functions, and a network to follow the information flow.
- Biological plausibility: controversial, though a local suboptimal approximation of BP has been supposed.

## Related

[[Neural Networks]], [[Training]], [[Delta Rule]], [[Notation]], [[Regularization]].
