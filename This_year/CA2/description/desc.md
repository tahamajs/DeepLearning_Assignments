In the Name of God

University of Tehran

Faculty of Electrical and Computer Engineering

Course: Deep Generative Models

Instructor: Dr. Mostafa Tavassoli

Homework Number: 1

Date: Mehr 1404 (October 2025)

---

### **Question 1: Probabilistic Graphical Models (PGM)**

#### **Part 1: Bayesian Networks Modeling**

Based on the following description, answer the questions below:

The occurrence and severity of a specific disease in an individual are affected by their Immune System (M). If the individual has a weak immune system, the Severity of the Disease (I) increases. Additionally, this disease is more severe during Cold Seasons (S). If the severity is high, an Expensive Medication (T) is recommended; otherwise, a Cheap Medication (T) is suggested. However, if the individual does not have the Financial Capability (F), they will use the cheap medication regardless. An expensive medication has a higher recovery rate and decreases the Probability of Death (D), whereas high disease severity increases the probability of death.

**Variables:**

- **M:** Immune System Strength
- **I:** Disease Severity
- **T:** Medication Used
- **S:** Season
- **F:** Financial Status
- **D:** Probability of Death (Outcome)

**Subsection 1 (3 Points):** Draw the **Bayesian Network (Graph)** representing the descriptions above.

**Subsection 2 (3 Points):** Write the **Joint Probability Distribution** based on the defined variables and the network.

**Subsection 3 (10 Points):** Based on your drawn graph, state whether the following statements are **True** or **False** and provide a brief reason (d-separation):

- a) **$F \perp D$**
- b) **$S \perp D \mid I$**
- c) **$M \perp F$**
- d) **$M \perp F \mid T$**
- e) **$M \perp T \mid \{D, I\}$**

---

#### **Part 2: Bayesian and Markov Graphs**

Answer the following questions based on the Bayesian Graph below:

**Subsection 1 (3 Points):** Write the **Joint Probability Distribution** of the variables according to the graph.

**Subsection 2 (1 Point):** Write the **Markov Blanket** for variable **T** .

**Subsection 3 (2 Points):** Is this graph a **Perfect I-Map** ? Why?

**Consider the following Markov Graph:**

**Subsection 4 (1 Point):** Is this graph **Chordal** ? Why?

**Subsection 5 (3 Points):** Identify the **Maximal Cliques** based on the Markov Graph and write the Joint Distribution based on these cliques.

**Subsection 6 (6 Points):** Show the correctness of these two statements:

- In a Bayesian Graph, to marginalize variable **$C$** (calculate joint over others), it is sufficient to simply remove the factor **$p(C)$**.
- In a Markov Graph, if **$\int \phi(C) dC = 1$**, it is sufficient to simply remove the factor **$\phi(C)$**.

---

#### **Part 3: Markov Properties**

Consider the following Markov Graph:

**Subsection 1 (3 Points):** Write the Joint Probability Distribution based on **Maximal Cliques** .

**Subsection 2 (8 Points):** State whether the following are **True** or **False** with reasoning:

- a) **$G \perp A$**
- b) **$F \perp A \mid \{D, C\}$**
- c) **$G \perp C \mid E$**
- d) **$p(A \mid B, C) = p(A \mid B, C, E)$**

**Subsection 3 (2 Points):** In the joint distribution from Subsection 1, if we multiply the potential function **$\phi(E, G)$** by 5, how does the final probability distribution change?

---

#### **Part 4: Variational Inference (10 Points)**

Consider a Bayesian Network with the following distributions for $z$ and $x$:

$p(z) = e^{-z}$ for $z > 0$

$p(x \mid z) = z e^{-zx}$ for $x > 0$

We want to approximate the posterior $p(z \mid x)$ using Variational Inference. We choose the following approximating distribution:

$q(z) = \theta^2 z e^{-\theta z}$ for $z > 0$

Given that $E_q[z] = \frac{2}{\theta}$, find the optimal value for the parameter $\theta$.

---

### **Question 2: Variational Autoencoders (VAE)**

#### **Part 1: Theory and Basic Implementation**

Subsection 1 (2 Points): The cost function for training a VAE is the ELBO (Equation 1). Explain why we cannot increase the data Likelihood directly and how the terms in this equation lead to model optimization.

$ELBO = E_{q(z|x)}[\ln p(x|z)] - D_{KL}(q(z|x) \parallel p(z))$

**Subsection 2 (1 Point):** Briefly explain the **dSprites** dataset used in this question and show a few samples.

**Subsection 3 (2 Points):** In VAEs, we sample from the latent distribution. This breaks the gradient chain. What is the solution to make the network end-to-end differentiable? Explain the **Reparameterization Trick** .

**Subsection 4 (10 Points):** Train a VAE model using the provided architecture (Table 1) and hyperparameters (Table 2).

- Plot the **Reconstruction Loss** , **KL Divergence** , and **Total Loss** during training.
- Pass 8 random samples through the model and compare the original vs. reconstructed versions.

Subsection 5 (2 Points): The $\beta$-VAE (Equation 2) was developed as an extension. What improvement does this model offer over the original VAE?

$Loss = E_{q(z|x)}[\ln p(x|z)] - \beta D_{KL}(q(z|x) \parallel p(z))$

**Subsection 6 (12 Points):** Train two **$\beta$**-VAE models with different **$\beta$** values (both **$>1$**). Plot their losses and compare their reconstruction results with a standard VAE (**$\beta=1$**).

**Subsection 7 (8 Points):** Explain the **MIG (Mutual Information Gap)** metric. Calculate this metric for your three models and compare the results.

**Subsection 8 (3 Points):** Perform **PCA** on the latent space components for one of your models. Plot the results and analyze if they align with the results from Subsection 7.

---

#### **Part 2: Advanced VAE Research (5 Points each)**

Explain the following concepts based on the provided references:

1. **VQ-VAE:** Explain the difference between this and the base model. What does "discretization" mean here?
2. **VampPrior:** How do the authors estimate the posterior, and how does this lead to better latent variable utilization compared to a standard normal prior?
3. **SC-VAE:** Explain this model and the role of the **ISTA** algorithm. What advantage does latent representation "sparsity" provide?

---

### **Submission Rules**

- **Deadline:** Tuesday, **13th of Aban** (November 4, 2025).
- **Grace Period:** Max 7 days after the deadline via the system (with penalty).
- **Submission:** One ZIP file named `HW1_[Lastname]_[StudentNumber].zip` containing the report (PDF) and code files.
- **Individual Work:** Plagiarism will result in a grade of zero for both parties.
- **Support:** For Q1, contact `ma.moghimi202@gmail.com`; for Q2, contact `S.m.moosavi000@ut.ac.ir`.
