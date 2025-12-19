# Deep Learning HomeWork 1

## Theory Part

### Problem 1: A Review to Linear Algebra

1. Prove that you can represent the Hessian matrix of a transformation like $y=\phi(u, v, z)$, as a gradient Jacobian matrix of this transformation. Assume u, v, z are singe-dimension, and y is a function of these three parameters.

2. Provide the proof for the following (Assume $y \in \mathbb{R}$, $x ∈ \mathbb{R}^n$. $a ∈ \mathbb{R}^n$, $X ∈ \mathbb{R}^{n*n}$, $A ∈ \mathbb{R}^{n*n}$)

    A. $\frac{∂ (x^⊤ a)}{∂ x} = \frac{\partial (a^\top x)}{∂ x} = a^\top$

    B. $\frac{∂ }{∂ y}tr(X) = tr(\frac{\partial }{∂  y}X)$

    C. $\frac{\partial}{∂ x}tr(X^⊤ AX) = X^⊤ (A + A^\top)$

    D. $\frac{\partial}{∂ x}log(det(X)) = X^{-\top}$

Hint: Remember that the inverse of a square matrix can be derived via the following formula:

$(X^{-1})_{ij} = \frac{1}{det(A)}C_{ij}$

3. The matrix for rotation in two dimensions, in $\theta$ degrees has the following form:

    $R(\theta) = \begin{bmatrix} \cos(\theta)  & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \\ \end{bmatrix}$

Find the eigenvalues, and the eigenvectors of the above matrix. Then, prove the following statement stands correct:

$det(X) = \prod_{i=1}^n \lvert \lambda_i \rvert$
($λ_i$ is an eigenvalue of matrix X.)

Finally, by making use of "Diagonalization of Eigenvalues", prove the following:

$R(n\theta) = R^n (\theta)$

### Problem 2: Optimization

1. Explain that why in higher dimensions, there are more saddle points than there are local minimum points.

2. Assume the optimization path is visualized for 4 methods "Nestrov-Momentum", "Momentum", "GD", "RMSprop", for a second order function from in interval [-2, +2].

   A. Explain what would be the optimization path curve look like, for each one of the methods.

   B. Explain about advantages, and disadvantages of all the mentioned methods, and discuss how the problems with GD, is solved by each one of the 3 other methods.

3. Explain how the ADAM method solves the problem with "Momentum" method, and why do we use "Bias Correction" in this approach.

### Problem 3: Regularization

Read the paper titled "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", by Srivastava, Hinton et al. Here is the abstract of the paper. The full text is freely accessible on the internet.

"Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique for addressing this problem. The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much. During training, dropout samples from an exponential number of different thinned networks. At test time, it is easy to approximate the effect of averaging the predictions of all these thinned networks by simply using a single unthinned network that has smaller weights. This significantly reduces overfitting and gives major improvements over other regularization methods. We show that dropout improves the performance of neural networks on supervised learning tasks in vision, speech recognition, document classification and computational biology, obtaining state-of-the-art results on many benchmark data sets."

After reading the paper, provide answer to questions 1 to 4.

1. Explain why Dropout has a performance similar to Ensemble-Learning, and why Dropout technique has better performance in networks with a high number of parameters?
2. Explain how Dropout technique, basically acts like a regularization technique, and is considered a regularization technique.
3. What is the difference between using Dropout technique during training vs during test? What is the reason behind this difference?
4. In the linear regression problem with N samples, with a data matrix $X ∈ \mathbb{R}^{N*D}$, and the target output $y ∈ \mathbb{R}^N$, and the Loss Function of $J(\omega) = \lVert y - X\omega \rVert_{2}^{2}$, if we apply Dropout technique to the input (i.e. Each element of matrix X will be present in regression problem with a probability $p$), Then, Prove that using Dropout technique is equivalent to using regularization term in the loss function. (Provide a comprehensive, detailed answer)
5. Explain why Batch-Normalization allows us to easily control learning rate of weights.
6. Discuss how Batch-Normalization results in regularization. With a solid reasoning, discuss the effect of larger batch size, on it's normalization property and characteristics.

### Problem 4: Activation Functions

1. For each one of the following classification problems, determine what is the proper activation function, for it's output layer. Discuss the reason behind your choice.

A. An image classifier that labels images either "Dog" or "Cat".

B. An image classifier that classifies which animal is in the input image. Assume there are 100 animals (classes).

C. An image classifier that determines what animals are present in an image. (There can be multiple animals in the image). This is a multi-class, multi-label classification problem.

2. ReLU is one of the commonly used activation functions to address the problem of vanishing gradients. However, the output of this function is zero, for inputs below zero. This results in some units not being trained and updated during training. For dealing with this problem, many variations of ReLU function have been proposed throughout the years. One of the most recent onces, is the DPReLU activation function. By referring to the paper that introduced this function, compare it with ReLU. Determine the rule of each parameter in this activation function. Finally, discuss what problems had the previous version of the DPReLU function, and how this function has been able to improve, or overcome those problems.

The paper you are going to refer to, is named "DPReLU: Dynamic Parametric Rectified Linear Unit and Its Proper Weight Initialization Method", published in 2023, in International Journal of Computational Intelligence Systems. The paper is open access and you can access to the full text freely. Here is the abstract of the said paper:

"Activation functions are essential in deep learning, and the rectified linear unit (ReLU) is the most widely used activation function to solve the vanishing gradient problem. However, owing to the dying ReLU problem and bias shift effect, deep learning models using ReLU cannot exploit the potential benefits of negative values. Numerous ReLU variants have been proposed to address this issue. In this study, we propose Dynamic Parametric ReLU (DPReLU), which can dynamically control the overall functional shape of ReLU with four learnable parameters. The parameters of DPReLU are determined by training rather than by humans, thereby making the formulation more suitable and flexible for each model and dataset. Furthermore, we propose an appropriate and robust weight initialization method for DPReLU. To evaluate DPReLU and its weight initialization method, we performed two experiments on various image datasets: one using an autoencoder for image generation and the other using the ResNet50 for image classification. The results show that DPReLU and our weight initialization method provide faster convergence and better accuracy than the original ReLU and the previous ReLU variants."

### Problem 5: Neural Networks and Backpropagation

Consider the following architecture, for a two layered neural network:

```
Nodes: 
  - Input: x1, x2
  - Hidden Layer: h1, h2
  - Output: y

Edges: 
  w1: (x1 -> h1)
  w2: (x1 -> h2)
  w3: (x2 -> h1)
  w4: (x2 -> h2)
  b1: (-> h1)
  b2: (-> h2)
  w5: (h1 -> y)
  w6: (h2 -> y)
```

Assume that the activation function of the hidden layer is LreLU, and the output activation function is sigmoid. Also, here are the values for weights, biases, inputs and Loss function:

- $[x_1, x_2] = [0, 1]$
- $[y] = [1]$
- $[w_1, w_2, w_3, w_4] = [0.3, 0.2, 0.2, -0.6]$
- $[w_5, w_6] = [0.5, -1]$
- $[b1, b2] = [0.2, -1.4]$
- $LreLU(x) = \{ 1 \lvert x \ge 0 , otherwise: 0.2x \lvert x \lt 0\}$
- $L = \frac{1}{2}(\hat{y} - y)^2$

1. Calculate Backpropagation for one step. Assume the loss function to be MSE, and the learning rate ($\alpha$) to be 0.1.

2. The batch normalization technique, which can be used in various layers of a network, by normalizing data belonging to a batch in a layer, helps the performance of the model on new data. assume $[x_1, x_2]$ is given as the input of the batch normalization layer, and intermittent variables $[\hat{x_1}, \hat{x_2}]$ are produced, which then are used in calculation of the output of the layer with the equation $\hat{y_k} = γ\hat{x_k} + β$. By describing the computational graph of the batch normalization layer in details, write the relation between mean and variance given that we have the input and output of the layer. (The learnable parameters of the layer are $\beta$ and $\gamma$.) Finally, calculate the partial derivative equations of the Loss function of network, with respect to $\beta, \gamma, x_1, x_2$,  assuming that we have the values for $\frac{\partial L}{∂ y_1}$, and $\frac{\partial L}{∂ y_2}$.
