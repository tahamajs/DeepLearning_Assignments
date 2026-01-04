

# Assignment 5: Neural Networks and Deep Learning

**University of Tehran - Faculty of Electrical and Computer Engineering**

* **Instructors:** Dr. Mohammad Gorji, Aryan Firouzi
* **Deadline:** Dey 16, 1404 (Approx. January 6, 2026)
* **Topics:** Image Re-identification (Transformers) & Large Language Diffusion Models

---

## Question 1: Image Re-identification using Transformers (Total: 100 Points + Bonus)

In this exercise, we will use Transformers for the task of Image Re-identification (Re-ID). Unlike traditional classification (e.g., distinguishing a cat from a dog), Re-ID involves identifying a specific instance (e.g., a specific person) across different camera views. This requires the network to learn fine-grained details (like tattoos, logos, or specific clothing patterns) rather than just general class structures.

We will compare two common architectures: **ResNet (CNN)** and **BotNet (Transformer-based)**.

### 1.1 Data Preparation (10 Points)

* Download the dataset from the provided link [Dataset Link].
* **Preprocessing:** Explain what preprocessing steps are useful for this dataset (e.g., resizing, normalization). Since we have fewer than 5 images per class, aggressive augmentation is necessary to increase accuracy. Explain which augmentation methods you used.
* **Splitting:** Split the dataset into **Train** and **Test** sets. You can use an 80/20 split (Train/Test).
* **Graph Sampling:** The text mentions using a "Graph Sampling" method (likely referencing a specific technique from the provided paper/resources). Explain how this sampling affects the data distribution.
* **Batch Size:** Use a batch size of 32 (or a reasonable number your hardware can handle).
* **Visualization:** Display a few sample images from the dataset after preprocessing.

### 1.2 ResNet Model Training (10 Points)

* Train a **ResNet50** model on the dataset.
* **Configuration:** Train for at least **20 epochs**. Use Cross-Entropy Loss.
* **Reporting:**
* Plot the Training Loss and Test Loss per epoch.
* Report the final accuracy/error rate on the Test set.
* There is no need to design the ResNet architecture from scratch; you may use existing libraries/implementations, but you must modify the output layer to match the number of classes in your dataset.



### 1.3 BotNet Model Training (40 Points)

* Implement and train a **BotNet** (Bottleneck Transformer) model.
* **Architecture:** BotNet replaces the spatial convolutions in the final stage (e.g., the last 3 layers of a ResNet) with **Multi-Head Self-Attention (MHSA)**.
* **Implementation:** You need to design the architecture such that the feature map dimensions are preserved while incorporating the attention mechanism.
* **Training:** Train this model using the same hyperparameters and data split as the ResNet model for a fair comparison.
* **Reporting:** Plot loss curves and report final accuracy.
* **Note:** Training might be slow (approx. 3-5 minutes per epoch on a T4 GPU in Google Colab). Ensure you use a GPU.

### 1.4 Results Analysis (15 Points)

* Compare the performance of ResNet and BotNet.
* Which model performed better? Why?
* Did the Transformer-based model (BotNet) perform better on specific classes or difficult cases compared to the CNN (ResNet)?
* Analyze if there is a significant difference in convergence speed or stability.

### 1.5 Attention Map Visualization & Analysis (15 Points)

* Extract the output of the **Attention Layer** from the trained BotNet.
* Visualize these outputs as **Heatmaps** overlaid on the original images (see Figure 1 in the original document as an example).
* **Analysis:**
* Does the model focus on the correct features (e.g., the person/object)?
* Is the model "cheating" by focusing on background information or irrelevant patterns?
* Compare the heatmaps of correct predictions vs. incorrect predictions.



### 1.6 Counterfactual Attention (Bonus: 5 Points)

* Implement **Counterfactual Attention** training.
* **Method:**
1. Get the standard output features from the model.
2. Mask the features based on their attention weights (masking the most attended/important features).
3. Calculate the entropy difference between the original output and the masked output.
4. Retrain/Fine-tune the model to maximize the entropy difference (forcing the model to look at other useful features, not just the most obvious ones).


* Repeat the analysis from section 1.5 with this new model. Did the heatmaps change?

---

## Question 2: Large Language Diffusion Models (Total: 100 Points + Bonus)

This question focuses on **LLaDA (Large Language Diffusion with Autoregression)**. Unlike standard Autoregressive (AR) models that predict the next token `t` based on `t-1` (left-to-right), LLaDA treats text generation as a **Masked Iterative Generation** process (similar to diffusion in image generation).

We will use this concept to build a **Text-to-SQL** pipeline.

### 2.1 Theoretical Questions (30 Points)

Answer the following questions based on the LLaDA paper and theory:

1. **Difference between Methodologies:** Explain the fundamental difference between **Autoregressive** generation (Next-token prediction) and **Masked Iterative Generation**. How does the generation process look different to the user?
2. **Forward Masking as Noise:** Explain how "Forward Masking" acts as a **Noise Model**. How does masking tokens relate to adding noise in continuous diffusion models?
3. **Reweighting Loss:** Why do we need to **reweight** the loss function?
* *Hint:* If we mask 15% of tokens vs. 90% of tokens, the information available to the model is different. Explain why we need a coefficient to balance the loss contribution of tokens masked at different probabilities.



### 2.2 Data, Prompt, and Evaluation Tools (20 Points)

**Dataset:** Use `gretelai/synthetic_text_to_sql` from Hugging Face.

#### 2.2.1 Pipeline Construction

* Load the dataset.
* Perform Train/Validation splits.
* **Stats:** Report the average length of the schema and questions (in tokens or characters).
* **Filtering:** If schemas are too long, truncation might cut off important table definitions. Implement a strategy to handle long schemas (e.g., keep only table names/columns and remove extra descriptions) or filter out examples that exceed `MAX_LEN`. Explain your decision in the report.
* **Requirements:** Use at least 1000 training samples and 300 test samples.

#### 2.2.2 Chat Format Prompt Design

* Design a specific prompt template using `tokenizer.apply_chat_template`.
* **Structure:**
* **System:** "You are a Text-to-SQL assistant. Output ONLY the SQL query. Do not add explanations."
* **User:** Contains the `Schema` and the `Question`.
* **Assistant:** Contains the `Gold SQL` (the correct answer).


* **Code Example:**
```python
SYSTEM_PROMPT = (
"You are a Text-to-SQL assistant. Output ONLY the SQL query. "
"Do not add explanations."
)
user_content = (
"Schema:\n"
"table students(id, name, age)\n"
"table enrollments(student_id, course_id)\n\n"
"Question:\n"
"List the names of students older than 20.\n\n"
)
messages_train = [
{"role": "system", "content": SYSTEM_PROMPT},
{"role": "user", "content": user_content},
{"role": "assistant", "content": "SELECT name FROM students WHERE age>20;"},
]
# Use tokenizer to format this

```


* **Crucial:** Ensure you know exactly where the "Assistant" response begins, as we will only apply noise/masking to the *Answer*, not the Prompt.

#### 2.2.3 SQL Normalization & Metrics

* Implement a **SQL Normalization** function:
* Lowercase the query.
* Remove backticks and quotes.
* Standardize spaces.
* Remove trailing semicolons `;`.


* **Metrics:** Implement **Exact Match (EM)** and **Normalized Exact Match**.
* *Note:* Two queries can be syntactically different but semantically identical (e.g., different whitespace). Normalized EM helps fix this.



### 2.3 Model Loading and Training (30 Points)

#### 2.3.1 Model Loading & Concepts

* Load the model `GSAI-ML/LLaDA-8B-Instruct`.
* **Quantization:** Use **4-bit quantization** to save memory. Explain what 4-bit quantization is and why it helps.
* **LoRA (Low-Rank Adaptation):** Use LoRA (via PEFT library) for fine-tuning. Explain what LoRA is and why we use it instead of full fine-tuning.
* **KV Cache:** Disable `use_cache=False` (since diffusion generation is not strictly left-to-right autoregressive in the training phase). Explain what KV Cache is typically used for.
* **Tokenizer:** Ensure `mask_token_id`, `pad_token_id`, and `eos_token_id` are set correctly.

#### 2.3.2 Forward Masking Implementation

Implement a function `noisy_batch` that takes a batch of `input_ids`:

1. Sample a random time step `t` (between 0 and 1) for each sample.
2. Based on `t`, determine the probability `p_mask` (e.g., if `t` is high, mask more tokens).
3. **Masking:** Randomly replace tokens in the **Assistant Response** part of the sequence with `[MASK]`.
4. **Constraints:** Ensure you do not mask the **Prompt** (System + User instructions).
5. **Output:** Return `masked_input_ids`, `masked_indices` (boolean mask showing where masks are), and `p_mask`.

#### 2.3.3 Boundary Handling

* You must identify the boundary between the `prompt` and the `answer`.
* Strategy: Tokenize `prompt` separate from `answer`, then concatenate.
* Only apply the masking logic to the `answer` portion indices.

#### 2.3.4 Loss and Training Loop

* **Loss Calculation:** Calculate Cross-Entropy Loss **only** on the tokens that were masked. The model should predict the original token at the masked positions.
* **Reweighting:** Apply the reweighting coefficient (discussed in 2.1.3) to the loss based on `p_mask`.
* **Normalization:** Decide how to normalize the loss (by total batch size vs. by number of masked tokens). Explain your choice.
* **Stability:** Handle edge cases (e.g., if no tokens are masked in a batch) to prevent `NaN` or crashes.
* **Optimization:** Use Gradient Clipping and a Learning Rate Scheduler.

### 2.4 Evaluation and Block Diffusion Sampling (20 Points)

#### 2.4.1 Block Diffusion Sampling

Implement the generation strategy used in LLaDA:

1. Start with a fully masked response (length `gen_length`).
2. **Iterative Process:**
* Feed the sequence (Prompt + Masked Response) to the model.
* The model predicts logits for the masked positions.
* **Block Selection:** Instead of accepting all predictions, select a "Block" of tokens (based on confidence or fixed block size) to **commit** (unmask).
* Leave the rest as `[MASK]` and feed it back into the model.
* Repeat until all masks are filled or `gen_length` is reached.


3. Explain why "Block" sampling is used (Efficiency vs. Accuracy trade-off).

#### 2.4.2 Post-processing

* Extract the SQL from the generated text.
* Clean up the output (remove text before `SELECT`, remove text after `;`).

#### 2.4.3 Error Analysis

* Evaluate on the test set (min 200 samples).
* Report **Normalized Exact Match**.
* Show 10 sample outputs (Prompt -> Generated SQL -> Ground Truth).
* Analyze common errors (e.g., hallucinating columns, wrong syntax).

#### 2.4.4 Bonus Idea (5 Points)

* Propose a creative idea to improve the LLaDA approach for Text-to-SQL (e.g., a change to the noise schedule, a different sampling strategy, or a specific architectural change). Explain the logic in a short paragraph.