# Multivariate Derivatives

## Directional and partial derivatives

The directional derivative of $f:\mathbb R^n\to\mathbb R$ at $x$ along $d$ is
$$
f'(x;d)=\lim_{t\downarrow0}\frac{f(x+td)-f(x)}{t}.
$$
Partial derivatives use coordinate directions $e_i$. Existence of all partial derivatives does **not** guarantee differentiability.

## Gradient and differentiability

$f$ is differentiable at $x$ if a vector $\nabla f(x)$ exists such that
$$
f(x+d)=f(x)+\nabla f(x)^Td+o(\|d\|).
$$
Then $f'(x;d)=\nabla f(x)^Td$. The gradient points in the direction of steepest increase under the Euclidean norm; $-\nabla f$ is steepest descent.

The level set $f(x)=c$ has tangent directions $d$ satisfying $\nabla f(x)^Td=0$, so the gradient is normal to a regular level set.

## Jacobian and chain rule

For $F:\mathbb R^n\to\mathbb R^m$, the Jacobian $J_F(x)\in\mathbb R^{m\times n}$ contains component derivatives. For $h(x)=g(F(x))$,
$$\nabla h(x)=J_F(x)^T\nabla g(F(x)).$$
This is the algebra behind backpropagation.

## Hessian

If $f$ is twice differentiable,
$$H_f(x)=\nabla^2f(x),\qquad (H_f)_{ij}=\frac{\partial^2f}{\partial x_i\partial x_j}.$$
Under continuous second partial derivatives it is symmetric. Curvature along $d$ is $d^TH_f(x)d$.

## Frequently used derivatives

$$
\nabla(a^Tx)=a,\qquad
\nabla\left(\frac12x^TQx\right)=\frac{Q+Q^T}{2}x,
$$
$$
\nabla\frac12\|Ax-b\|_2^2=A^T(Ax-b),\qquad
\nabla^2\frac12\|Ax-b\|_2^2=A^TA.
$$

See [[Taylor Models and Optimality Conditions]] and [[Neural Networks as Optimization Models]].


