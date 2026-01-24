
## University of Tehran — Faculty of Electrical Engineering

### Deep Learning and Neural Networks

**Exercise / Assignment 6**: Improving network performance by increasing distribution capacity with **VAE**, and improving **GAN** with limited data for augmentation

**Course staff (as shown in the document):**

* Instructor: *Mehdi Mousavi*
* Teaching Assistant: *Saeed Rahimi*
* (Other names appearing in the header were preserved as-is where readable.)

**Deadline (Solar Hijri):** 1404/10/29 (Dey 29, 1404)

---

# Question 1 — VAE (Total: 20 points)

> The referenced VAE paper is not only about reconstruction (like standard autoencoders). In particular, pay close attention to the **KL term** and how the paper proposes a modified objective to increase the model’s overall capacity/quality.
> The paper also proposes a **hierarchical architecture**; you are expected to implement that proposal.

### 1–1. Theory questions (20 points total)

#### 1–1–1. (10 points) Standard VAE loss function

Explain the **VAE objective** and the role of the **KL divergence** term. Clearly define:

* the encoder ( q_\phi(z \mid x) ),
* the decoder ( p_\theta(x \mid z) ),
* the prior ( p(z) ),
* the **ELBO** and how it relates to log-likelihood.

#### 1–1–2. (5 points) The “new” loss term in the paper

Explain what **new loss/term** the paper introduces compared to the standard VAE objective, and **why** it is introduced (what limitation it addresses).

#### 1–1–3. (5 points) Type of improvement from the new objective

Explain what kind of improvement the modified loss is intended to deliver (e.g., higher capacity, better sample quality, better latent utilization, mitigating posterior collapse, etc.). Your explanation must be tied to the paper’s logic (not generic claims).

---

### 1–2. (10 points) Dataset familiarity: Dynamic MNIST

Introduce **Dynamic MNIST** and explain:

* how it differs from “standard/binarized MNIST” usage,
* how samples are generated/processed per iteration (dynamic binarization),
* why it is commonly used in likelihood-based generative modeling.

---

### 1–3. (10 points) Implement the “low-dimensional latent” model

Implement the model described in the paper using a **low-dimensional latent space** (as specified in the assignment/paper).
Report:

* training curves (loss terms separately),
* qualitative samples.

---

### 1–4. (15 points) Analyze latent distribution behavior

For **both** models (baseline VAE vs the paper’s proposed model), analyze:

* the learned posterior ( q_\phi(z \mid x) ) statistics,
* how close it is to the prior ( p(z) ),
* any visible structure in latent space.

---

### 1–5. (10 points) Implement the “high-dimensional latent” model

Implement the paper’s model using a **higher-dimensional latent** (the assignment text mentions “suggested 40” for the larger latent; use whatever the paper/assignment specifies).
Compare with the low-dimensional case:

* loss curves,
* sample quality,
* latent utilization indicators.

---

### 1–6. (10 points) Study **posterior collapse**

Explain what posterior collapse is, why it happens in VAEs, and **compare** the two models with respect to:

* latent activity,
* KL term behavior,
* decoder dominance symptoms.

---

### 1–7. (10 points) Latent space exploration / traversal

Perform latent traversals and/or PCA on latent codes and report:

* whether principal directions are meaningful,
* whether class/semantic structure emerges,
* how this differs between the two models.

(Your pasted text explicitly mentions **PCA** and checking whether results are “meaningful.”)

---

### 1–8. (10 points) **Pseudo-inputs** (Pseudo-Inputs) visualization

Show and interpret pseudo-inputs and explain their role (as used in the referenced method). Provide visualizations and a brief analysis.

---

### 1–9. (10 points) Log-likelihood computation

Compute/estimate **log-likelihood** as requested in the assignment (the text explicitly asks for “log likelihood calculation”).
You must explain **which estimator** you use (e.g., importance sampling / IWAE-style estimate) and why naive likelihood is not directly tractable for latent-variable models.

---

# Question 2 — GAN for data augmentation with limited data

The assignment motivation (as stated): when data are limited, deep models can generalize poorly, class boundaries are learned incorrectly, and systematic errors can occur. **Data augmentation** is a standard remedy; **generative models (GANs)** can generate new samples to expand the dataset distribution.

You will implement a **conditional GAN** and use it for augmentation, then train a classifier and compare performance.

---

## 2–1. (5 points) Initial setup + dataloader construction

* Use **FashionMNIST**.
* Use the provided normalization:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
```

* Download dataset:

```python
from torchvision import datasets
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)
```

* Create **two non-overlapping subsets**:

  1. One subset for training a **classifier**
  2. One subset for training the **GAN**

The subsets must have **no overlap**.

The text indicates a per-class sampling requirement (readable parts show **500 per class** in at least one place; keep exactly what your assignment statement requires if it differs).

---

## 2–2. (5 points) Implement Channel Attention and Self Attention

Implement exactly as described in the paper/assignment:

* Provide classes named **ChannelAttention** and **SelfAttention**.
* For SelfAttention:

  * Produce tensors named **f, g, h, v** (as specified).
  * Attention map via **softmax**, with shape **(B, N, N)**.
  * Inputs/outputs follow the documented reshape rules ((N = H \times W)).
  * Use learnable scalar **(\gamma)** and residual form:
    [
    y = \gamma o + x
    ]
* For ChannelAttention:

  * Use **global average pooling**, then two fully-connected layers (with ReLU), then **sigmoid** to form channel weights, applied back to the feature map.

---

## 2–3. (5 points) Spectral Normalization for Conv layers

Implement **Spectral Normalization (SN)** for convolutional layers as described:

* Use a helper/wrapper so it can be applied to layers in both **Discriminator** and **Generator**.
* Ensure required Conv/ConvTranspose parameters exist:

  * Conv2d: `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `bias`
  * ConvTranspose2d: also includes `output_padding`

Also explain:

* what SN does (Lipschitz control via spectral norm of weight matrix),
* why it stabilizes GAN training.

---

## 2–4. (15 points) Implement Generator & Discriminator architecture (per the paper)

You must match:

* block ordering,
* channel counts,
* kernel sizes, strides, paddings,
* the exact placement of **SelfAttention** and **ChannelAttention**,
* application of **Spectral Normalization** on the correct layers.

### 2–4–1. Extract and list architecture details

Provide a clear description of:

* each layer/block,
* parameters (C, H, W evolution),
* where attentions are inserted.

### 2–4–2. Projection Discriminator + Conditional setting

Your GAN is **conditional**:

* Class label (y) is embedded into a vector (embedding dimension appears as **100** in your table).
* For **Generator**: concatenate noise (z) and label embedding.
* For **Discriminator**: use **Projection Discriminator**:

  * compute feature vector (\phi(x)),
  * compute label embedding (e(y)),
  * add an inner product ( \langle \phi(x), e(y) \rangle ) to the discriminator score.

The text notes a feature dimension of **512** and implies the embedding must align dimensionally.

---

## 2–5. (15 points) Implement WGAN-GP and full loss terms

Implement:

* **WGAN-GP** Wasserstein loss
* **Gradient Penalty** with (\lambda_{GP} = 10)
* **Drift penalty** with coefficient **0.001**
* **Embedding L2 regularization** with coefficient **0.001** (applied to discriminator’s class embedding weights)

Your pasted assignment already includes pseudocode for:

* `GRADIENT_PENALTY`
* `DISCRIMINATOR_LOSS`
* `GENERATOR_LOSS`

You should implement exactly that logic.

---

## 2–6. (10 points) Final tuning + hyperparameter setup

Follow the assignment’s explicit hyperparameters:

### Weight initialization

* Conv / Linear / Embedding: Normal(mean=0.0, std=0.02)
* BatchNorm weight: Normal(mean=1.0, std=0.02)
* All biases: 0.0
* If spectral norm exists: initialize `weight_orig` appropriately

### Weight decay grouping

* Embedding parameters: weight decay = **0.001**
* All non-embedding parameters: weight decay = **0.0**
* Embeddings exist in **Generator** and **Discriminator** as specified.

### Optimizers

* Adam for both G and D with:

  * **LR(G) = 0.0002**
  * **LR(D) = 0.0004**
  * betas: **(0.0, 0.9)**

### Scheduler

* Exponential LR decay with **gamma = 0.95**
* Apply decay **every 100 steps**

### Training constants

* `N_CRITIC = 5`
* Total steps: `RUN = 4000`
* `FIXED_PER_CLASS = 8`
* Noise distribution: Normal(0,1)

---

## 2–7. (10 points) Model training loop

Implement the training loop respecting:

* Update **Discriminator** `N_CRITIC` times per Generator update.
* Use the same labels for real and fake batches during critic updates (as stated).
* Ensure fake images are detached for D updates.
* Log all required terms (Wasserstein term, GP, drift, emb L2, total losses).

---

## 2–8. (10 points) Analysis + plots + generated samples

### 2–8–1. Plot curves (x-axis: step)

Plot:

* Discriminator loss
* Generator loss
* Wasserstein estimate
* Gradient penalty
* Drift penalty
  (Your text requests multiple curves; keep them clearly separated and labeled.)

### 2–8–2. Visual outputs over training

Generate images at steps:

* 0, 1000, 2000, 3000, 4000
  using:
* a **fixed noise** tensor,
* fixed labels (include all classes),
* consistent grid visualization.

Explain why fixed noise/labels are required for a fair comparison across training.

---

## 2–9 / 2–10 / 2–11. Classification evaluation + comparison (as implied)

The assignment text indicates:

* train a classifier baseline on limited real data,
* then train a classifier on (real + GAN-generated) augmented data,
* compare class-wise errors and overall accuracy,
* discuss whether GAN augmentation improves generalization and why.

---

# Scientifically standard references (highly relevant to this assignment)

If you are writing the report in a formal scientific style, these are the canonical sources typically cited:

* **VAE**: D. P. Kingma & M. Welling, *Auto-Encoding Variational Bayes*, 2014.
* **VAE (alternative derivation)**: D. J. Rezende et al., *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*, 2014.
* **Likelihood estimation / IWAE**: Y. Burda et al., *Importance Weighted Autoencoders*, 2016.
* **WGAN-GP**: I. Gulrajani et al., *Improved Training of Wasserstein GANs*, 2017.
* **Spectral Normalization**: T. Miyato et al., *Spectral Normalization for GANs*, 2018.
* **Projection Discriminator**: T. Miyato & M. Koyama, *cGANs with Projection Discriminator*, 2018.
* **Self-Attention GAN (if this is the paper you are matching)**: H. Zhang et al., *Self-Attention Generative Adversarial Networks*, 2019.
* **Channel attention (common formulation)**: J. Hu et al., *Squeeze-and-Excitation Networks*, 2018.
