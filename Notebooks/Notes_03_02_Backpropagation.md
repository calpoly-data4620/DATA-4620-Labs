# Notes on Backpropagation

## Jonathan Ventura

### Backpropagation

Consider a multi-layer perceptron with a single hidden layer with ReLU activation and a output layer size of 1.  Here is the step-by-step computation from input $\vec{x}$ to output $z$ and loss $L$:

$$\vec{h} = W^{(1)} \vec{x} + \vec{b}^{(1)}$$
$$\vec{s} = \textrm{ReLU}(\vec{h})$$
$$z = W^{(2)} \vec{s} + b^{(2)}$$
$$L = \frac{1}{2}(y-z)^2$$

This is called the "forward" step as the computation flows forward from the input to the output and then the loss.

Now we want to calculate the derivatives of the loss $L$ w.r.t. the weights $W^{(1)},\vec{b}^{(1)},W^{(2)},$ and $b^{(2)}$.  This is called the "backward" step as we need to calculate the derivatives starting from the loss and moving backward toward the input.

To calculate these derivatives, we first need to understand the concepts of the Jacobian matrix and the generalized chain rule.

### Jacobian matrix

Consider a function $g : \mathbb{R}^n : \mathbb{R}^m$.  We will write $g_i(\vec{x})$ to mean the $i$-th element of $g(\vec{x})$ and $x_j$ to mean the $j$-th element of $\vec{x}$.   The $m \times n$ matrix $J$ of all partial derivatives of $g(\vec{x})$ is called the **Jacobian**:

$$J  =
\begin{bmatrix}
\frac{\partial{g_1}}{\partial{x_1}} & \cdots & \frac{\partial{g_1}}{\partial{x_n}} \\
\vdots \\
\frac{\partial{g_m}}{\partial{x_1}} & \cdots & \frac{\partial{g_m}}{\partial{x_n}} \\
\end{bmatrix}
$$

Note that the Jacobian matrix has size $m \times n$ where $m$ is the number of outputs and $n$ is the number of inputs.  The rows of the Jacobian correspond to the outputs, and the columns correspond to the inputs.

As a shorthand, I will indicate the Jacobian using the partial derivative notation, so that the Jacobian of $g(\vec{x})$ is $\partial{g}/\partial{\vec{x}}$.

### Generalized chain rule 

The generalized chain rule applies when computing partial derivatives of a function with a vector input and output.  Consider a function $f : \mathbb{R}^m \rightarrow \mathbb{R}$, function $g : \mathbb{R}^n \rightarrow \mathbb{R}^m$, and input vector $\vec{x}$ of size $n \times 1$.  The generalized chain rules states that:

$$\frac{\partial f}{\partial x_j} = \sum_{i=1}^m \frac{\partial f}{\partial g_i}\frac{\partial g_i}{\partial x_j}$$

Written as a matrix multiplication, we have:

$$\frac{\partial f}{\partial x_j} =
\begin{bmatrix}
\frac{\partial f}{\partial g_1} & \cdots & \frac{\partial f}{\partial g_m}
\end{bmatrix}
\begin{bmatrix}
\frac{\partial g_1}{\partial x_j} \\ \vdots \\ \frac{\partial g_m}{\partial x_j}
\end{bmatrix}
$$

$$=\frac{\partial f}{\partial g} 
\frac{\partial g}{\partial x_j}.
$$

The Jacobian matrix $\partial f / \partial \vec{x}$ is thus:

$$\frac{\partial f}{\partial \vec{x}}
= \begin{bmatrix}
\frac{\partial f}{\partial x_1} & \cdots & \frac{\partial f}{\partial x_m}
\end{bmatrix}
$$
$$= \begin{bmatrix}
\frac{\partial f}{\partial g_1} & \cdots  \frac{\partial f}{\partial g_n}
\end{bmatrix}
\begin{bmatrix}
\frac{\partial g_1}{\partial x_1} & \cdots & \frac{\partial g_1}{\partial x_m} \\
& \ddots &\\
\frac{\partial g_n}{\partial x_1} & \cdots & \frac{\partial g_n}{\partial x_m}
\end{bmatrix}
$$

$$=
\frac{\partial f}{\partial g} 
\frac{\partial g}{\partial \vec{x}}.
$$

### Derivatives of common neural network operations

#### Loss function

The derivative of the squared error loss function is simple to derive:

$$L(z) = \frac{1}{2}(z-y)^2$$
$$\frac{dL}{dz} = (z-y)$$

#### Activation function

The ReLU activation function is written as follows:

$$\textrm{ReLU}(x) = \min(0,x)$$

Clearly $\partial{\textrm{ReLU}}/\partial x = 0$ when $x<0$ and 1 when $x>0$.

What about when $x=0$?  ReLU is not differentiable at this point, but we can simply choose the slope to be zero (or one).  (See [Deep Learning $\S$6.3](https://www.deeplearningbook.org/contents/mlp.html) for more discussion on this point.). In summary we have:

$$\frac{\partial{\textrm{ReLU}}}{\partial x} = \begin{cases}
0 && \textrm{if } x \leq 0 \\
1 && \textrm{if } x > 0.
\end{cases}
$$

Another way of writing this is $\partial{\textrm{ReLU}}/\partial{x}=[x>0].$

When we apply the activation function to a vector $\vec{x}$, so that we have $\vec{s}=\textrm{ReLU}(\vec{x})$, ReLU is simply applied to each element in the same way: $s_i =\textrm{ReLU}(x_i)$ for $i=1,\ldots,n$.  Therefore the Jacobian is a diagonal matrix of ones and zeros:

$$\frac{\partial{\vec{s}}}{\partial \vec{x}} = 
\begin{bmatrix}
[x_1>0] & & & \\
& [x_2>0] & & \\
& & \ddots & \\
& & & [x_n>0]
\end{bmatrix}
$$

Suppose we have already calculated $\partial{L}/\partial{\vec{s}}$ and we now want to calculate $\partial{L}/\partial{\vec{x}}$.  By the chain rule we have

$$\frac{\partial{L}}{\partial{\vec{x}}}=
\frac{\partial{L}}{\partial{\vec{s}}}
\frac{\partial{\vec{s}}}{\partial{\vec{x}}}
$$
$$
=\begin{bmatrix}
\frac{\partial{L}}{\partial{s_1}} & \cdots  & \frac{\partial{L}}{\partial{s_n}}
\end{bmatrix}
\begin{bmatrix}
[x_1>0] & & & \\
& [x_2>0] & & \\
& & \ddots & \\
& & & [x_n>0]
\end{bmatrix}
$$
$$
=\begin{bmatrix}
\frac{\partial{L}}{\partial{s_1}}[x_1>0] & \cdots  & \frac{\partial{L}}{\partial{s_n}}[x_n>0]
\end{bmatrix}
$$

So we see that all we need to is zero out $\partial{L}/\partial{\vec{s}}$ wherever $\vec{x}\leq0$.

#### Affine transformation

Now consider the affine transformation, the building block of the multi-layer perceptron:

$$\vec{z} = W\vec{x}+\vec{b}$$

First, let's look at how each element of $\vec{z}$ is computed:

$$z_i = \sum_{k=1}^{n} W_{ik} x_k + b_i.$$

We will first consider $\partial{\vec{z}}/\partial{\vec{b}}$.  Clearly for each element $z_i$, the derivative with respect to $b_i$ is one and the derivative with respect to all other elements of $\vec{b}$ is zero.  So we simply have the identity matrix:

$$\frac{\partial{\vec{z}}}{\partial{\vec{b}}} = I.$$

Now for the Jacobian w.r.t. $\vec{x}$.  We see that $\partial{z_i}/\partial{x_j} = W_{ij}$ and it follows that

$$\partial{\vec{z}}/\partial \vec{x} = W.$$

Finally we consider $\partial{\vec{z}}/\partial{W}$.  Looking at the computation of $z_i$, we see that

$$
\frac{\partial{z_i}}{\partial{w_{jk}}}=
\begin{cases}
x_k & \textrm{if } i=j \\
0 & \textrm{otherwise.}
\end{cases}
$$

Let $W_j$ be the $j$-th row of $W$. It follows that

$$
\frac{\partial{z_i}}{\partial{W_j}}=
\begin{cases}
\vec{x}^T & \textrm{if } i=j \\
0 & \textrm{otherwise}
\end{cases}.
$$

If we are aiming to compute the Jacobian $\partial L/\partial W$ for the loss function $L(\vec{z})$, we see that the computation collapses nicely into an outer product of two vectors. First looking at the Jacobian w.r.t $W_j$:

$$\frac{\partial{L}}{\partial{W_j}} =\sum_{i=1}^{m} 
\frac{\partial{L}}{\partial{z_i}}
\frac{\partial{z_i}}{W_j}
=\frac{\partial{L}}{\partial{z_j}}
\frac{\partial{z_j}}{W_j}
=\frac{\partial{L}}{\partial{z_j}}
\vec{x}^T.
$$

Now we put the rows together to form $\frac{\partial L}{\partial W}$:

$$
\frac{\partial L}{\partial W}
=\begin{bmatrix}
\frac{\partial{L}}{\partial{W_1}} \\
\vdots \\
\frac{\partial{L}}{\partial{W_m}}
\end{bmatrix}
=\begin{bmatrix}
\frac{\partial{L}}{\partial{z_1}}
\vec{x}^T \\
\vdots \\
\frac{\partial{L}}{\partial{z_m}}
\vec{x}^T
\end{bmatrix}$$

$$=\left(\frac{\partial{L}}{\partial{\vec{z}}}\right)^T\vec{x}^T.$$ 

#### Accumulating gradients for backpropagation

Coming back to our original problem of calculating the gradients necessary to update our network weights, we see that we can calculate them by post-multiplying a chain of Jacobian matrices togeher.

For example, to calculate $\partial L/\partial W^{(2)}$ and $\partial L/\partial b^{(2)}$:

$$\frac{\partial L}{W^{(2)}} = \frac{\partial L}{\partial z}\frac{\partial z}{\partial W^{(2)}}$$
$$\frac{\partial L}{b^{(2)}} = \frac{\partial L}{\partial z}\frac{\partial z}{\partial b^{(2)}}.$$

Note that we can compute $\partial L / \partial z$ once and reuse it for both computations.

As we move backward, we can continue to update the loss gradient through the chain rule:

$$\frac{\partial L}{\partial \vec{s}} = 
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial \vec{s}}$$

$$\frac{\partial L}{\partial \vec{h}} = 
\frac{\partial L}{\partial \vec{s}}
\frac{\partial \vec{s}}{\partial \vec{h}}$$


Finally we arrive at the Jacobians for $W^{(1)}$ and $\vec{b}^{(1)}$:

$$\frac{\partial L}{\partial W^{(1)}} = 
\frac{\partial L}{\partial \vec{h}}
\frac{\partial \vec{h}}{\partial W^{(1)}}
$$

$$\frac{\partial L}{\partial \vec{b}^{(1)}} = 
\frac{\partial L}{\partial \vec{h}}
\frac{\partial \vec{h}}{\partial  \vec{b}^{(1)}}
$$

As explained above, in practice some of these computations are not implemented by constructing and multiplying the full Jacobian matrices, but instead performed in a more efficient manner depending on the sparsity pattern of the Jacobian.
