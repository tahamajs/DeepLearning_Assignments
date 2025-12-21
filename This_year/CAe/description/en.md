
# University of Tehran

**Faculty of Electrical & Computer Engineering**
**Course:** Deep Learning and Neural Networks
**Assignment:** Homework Projects

**Design Team:** Babak Mohtasham, Ranjbar, Hosseini, Javad, Mohammad Ghaderi, Yousef
**Reviewers:** Komijani, Mirhosseini
**Deadline:** January 18, 2026 (Dey 28, 1404)

---

## Question 1: Generating Text Descriptions for Images (Image Captioning)

**Goal:** In this question, we will help visually impaired individuals or generate subtitles for videos/images using Deep Learning. We will combine Computer Vision (for images) and Natural Language Processing (for text) to generate human-like textual descriptions for images. This task is known as **Image Captioning**.

### 1.1 Data Preparation (15 Points)

1. **Download:** Download the provided dataset (Flickr/COCO subset). The dataset includes an `images` folder and a CSV file containing image-to-text translations (Persian descriptions).
2. **Preprocessing:**
* You must preprocess the text.
* Create a **Vocabulary** (Dictionary) from the training data. This dictionary should include all unique words found in the dataset.
* Include 4 special tokens in your dictionary: `<start>` (sos), `<end>` (eos), `<pad>`, and `<unk>` (unknown). Explain the importance of each.


3. **Tokenizer:**
* Implement a `Tokenizer` class.
* This class should convert sentences into a list of tokens (indices) based on your vocabulary.
* Handle emojis and remove punctuation.
* Implement a function to reverse the process (convert indices back to text).


4. **Data Loader:**
* Create a PyTorch `Dataset` and `DataLoader`.
* Analyze the statistical distribution of caption lengths (min, max, average).
* Implement a `collate_fn` to pad sentences to a fixed length (or the max length in a batch) using the `<pad>` token.
* The dataloader should return: The image, the caption (tokenized), and the caption length.



### 1.2 Model Implementation

#### 1.2.1 Image Processing (Encoder) (5 Points)

* Use a pre-trained CNN model (e.g., **VGG16**) trained on ImageNet.
* Remove the final classification layers.
* **Freeze** the weights of the CNN feature extractor to save training time.
* Pass the image through the CNN to extract feature vectors.

#### 1.2.2 Text Generation (Decoder) (15 Points)

* Implement a Decoder using an **LSTM** architecture.
* **Attention Mechanism:** Incorporate an Attention mechanism (as introduced in the "Show, Attend and Tell" paper). Explain the difference between a standard LSTM decoder and one with Attention.
* **Teacher Forcing:**
* Explain the concept of Teacher Forcing (using ground truth as input for the next step during training vs. using the model's own prediction).
* **Figure 1:** Refer to the diagram (implied) showing Teacher Forcing vs. non-Teacher Forcing.
* **Implementation:** Implement the training loop using Teacher Forcing (feeding the correct token from the dataset).
* **Inference:** During evaluation/testing, Teacher Forcing cannot be used. You must implement a **Greedy Search** strategy (taking the token with the highest probability at each step) to generate the full sentence until the `<eos>` token is reached.


* **Dimensions:** Use an embedding size of 300 and hidden size of 512.

### 1.3 Training and Evaluation

#### 1.3.1 Model Training (10 Points)

* Train the model using the **Cross Entropy Loss** function.
* Use the training set and validate using the validation set.
* Save the best model based on the lowest validation loss.
* Plot the loss curve for each epoch.

#### 1.3.2 Model Evaluation (15 Points)

* **Qualitative:** Generate captions for 5 random images from the test set. Analyze the results. Are the errors reasonable?
* **Quantitative:** Calculate **BLEU Scores** (BLEU-1 to BLEU-4). Explain how the BLEU metric works.
* **Search:** Implement a search function where, given a text query, the model retrieves the most relevant image (or vice versa, given the nature of the decoder).

### 1.4 Self-Supervised Learning (Contrastive - CLIP)

**Concept:** We will now implement a model inspired by **CLIP (Contrastive Language-Image Pre-training)**. The goal is to learn a joint embedding space for text and images where matching image-text pairs have high cosine similarity.

* **Loss Function:** Implement **InfoNCE Loss**. This loss maximizes the similarity between correct pairs (positive samples) and minimizes similarity with incorrect pairs (negative samples) in the batch.
* **Zero-Shot Classification:** Explain the concept of Zero-Shot classification using CLIP.

### 1.5 Implementation of CLIP

#### 1.5.1 Image Processing (ViT) (5 Points)

* Use the `timm` library to load a **Vision Transformer (ViT)** (e.g., `vit_small_patch16_224`).
* Code snippet provided:
```python
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

```


* Add a fully connected projection layer to map the ViT output to the desired embedding dimension. Normalize the output vector.

#### 1.5.2 Text Processing (5 Points)

* Use a simple text encoder (like the LSTM from Q1 or a small Transformer) to convert text into a feature vector.
* Add a projection layer to map the text features to the same dimension as the image features.

#### 1.5.3 Fusion (5 Points)

* Calculate the dot product between the image embeddings and text embeddings.
* Scale the result by a learnable **Temperature** parameter. Explain the effect of temperature.

### 1.6 Training (10 Points)

* Train the model using the InfoNCE loss on the image-text pairs.
* Plot the training and validation loss.
* **Investigate:** Should we freeze the ViT backbone or fine-tune it? Experiment or research and justify your choice.

### 1.7 Evaluation (15 Points)

* **Image Retrieval:** Given a text query, retrieve the top-5 most similar images (Top-5 Accuracy).
* **Text Retrieval:** Given an image, retrieve the top-5 most similar captions.
* Report Top-1, Top-5, and Top-10 accuracy.

---

## Question 2: Urban Sound Classification with Wav2Vec

**Goal:** Classify urban sounds (e.g., car horn, drilling, dog barking) using **Wav2Vec 2.0** and Self-Supervised Learning (SSL).

### 2.1 Theoretical Questions (30 Points)

#### 2.1.1 Performance of Self-Supervised Models

* **A)** Explain the training process of **HuBERT** and **Wav2Vec 2.0**. How do they differ? What is the role of the "Cluster" step in HuBERT?
* **B)** **Masking:** Explain the role of Masking in Wav2Vec 2.0. How does the Contrastive Loss function work here?
* **C)** Compare the approach of using **Raw Waveform** (Wav2Vec) vs. **Mel-Spectrogram + CNN**. What are the pros and cons of each?

#### 2.1.2 Audio Data Augmentation & Evaluation

* **A)** Name and explain 3 common **Time Domain** augmentation techniques (e.g., Time Shift, Noise Injection). Why are they important for robustness?
* **B)** **Hand-Engineered Features:** Name 3 common features (e.g., MFCC, Pitch, Zero Crossing Rate). Explain why "Pitch" might be useful for gender recognition but less critical for Automatic Speech Recognition (ASR).

### 2.2 Model Training and Evaluation

**Dataset:** **UrbanSound8K**.

* 8732 labeled sound clips (<= 4 seconds) from 10 classes (air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music).

#### 2.2.1 Training Strategies (Implementation)

You will compare a simple **CNN** (trained on Spectrograms) vs. **Wav2Vec 2.0**.

* **Data Split:** Train: 80%, Val: 10%, Test: 10%.
* **Hyperparameters:** Epochs=5, Learning Rate=1e-4, Optimizer=AdamW.

**Experiments (Table 2):**

1. **Simple CNN:** Train from scratch.
2. **Wav2Vec (Pre-trained):** Load `facebook/wav2vec2-base`.
* **Freeze All:** Freeze the Wav2Vec backbone, train only the classification head.
* **Partial Freeze:** Freeze the first 6 layers of the Transformer, train the rest.
* **Full Fine-Tuning:** Unfreeze all parameters.



**Analysis:**

* Compare the convergence speed and final accuracy.
* Why does Transfer Learning (Wav2Vec) generally perform better than training from scratch?
* Analyze the effect of freezing layers.

#### 2.2.2 Evaluation

* Report **Accuracy, F1-Score (Macro)**, and **Confusion Matrix** for all models.
* Plot Loss and Accuracy curves.
* **Conclusion:** Which strategy (Freezing vs. Fine-tuning) is best for this specific dataset size?

---

## Question 3: Fine-tuning LLM with LoRA for Sentiment Analysis

**Goal:** Fine-tune a Large Language Model (**Llama-3.2-1B**) for sentiment analysis using **LoRA (Low-Rank Adaptation)** on the **Emotion** dataset.

### 3.1 Concepts (10 Points)

* **Full Fine-Tuning vs. LoRA:** Define both.
* **LoRA Mechanism:** Explain how LoRA injects trainable low-rank matrices ( and ) into the frozen pre-trained weights (). .
* Explain the advantages of LoRA regarding memory efficiency and storage.

### 3.2 Hugging Face & Datasets

#### 3.2.1 Familiarity with HF

* **Dataset:** We will use the `emotion` dataset from Hugging Face.
```python
from datasets import load_dataset
dataset = load_dataset('emotion')

```


* **Labels:** {0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise}.
* **Access:** Note that accessing Llama models requires accepting the license on Hugging Face and using an access token (`huggingface-cli login`).

### 3.3 Implementation

#### 3.3.1 Stratified Sampling (10 Points)

* Create a subset of the data to save time.
* Use **Stratified Sampling** to ensure the class distribution in your subset matches the original dataset.
* Train size: 1500, Test: 100, Validation: 50.



#### 3.3.2 Load Model & Tokenizer (10 Points)

* Load `meta-llama/Llama-3.2-1B` and its tokenizer.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

```



#### 3.3.3 Format Prompt (10 Points)

* Create a function `format_prompt` to structure the input for the LLM.
* **Format:**
```python
prompt = f"<s>[INST] {system_instruction} [/INST] {user_input} </s> [ASSISTANT] {assistant_output} </s>"

```


* **Example:** "Analyze the sentiment of the following text... [INST] i feel like i have been a neglectful lady [/INST] [ASSISTANT] sadness </s>"

#### 3.3.4 Tokenization & Encoding (10 Points)

* Tokenize the formatted prompts.
* Settings: `truncation=True`, `padding="max_length"`, `max_length=128`.
* Show one example of a tokenized sequence decoded back to text to verify correctness.

### 3.4 Training with LoRA (40 Points)

* **Configuration:**
```python
r_values = 8  # Rank
lora_alpha_values = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_dropout = 0.1

```


* Explain what `rank`, `alpha`, and `target_modules` do.
* **Training Arguments:**
* Use `cosine` learning rate scheduler.
* `fp16=True` (Mixed precision).
* `num_train_epochs=10`.


* **Train:** Train the model using the `Trainer` class or a custom loop.
* **Plot:** Plot the Training Loss.

### 3.5 Evaluation (10 Points)

* Compare three models:
1. **Base:** `meta-llama/Llama-3.2-1B` (Zero-shot).
2. **Instruct:** `meta-llama/Llama-3.2-1B-Instruct` (Zero-shot).
3. **Fine-tuned:** Your LoRA model.


* **Metrics:** Calculate **Accuracy** and **F1-Score** for all three on the test set.
* **Confusion Matrix:** Plot the confusion matrix for the best model.

---

## Question 4: Adversarial Attacks

**Goal:** Understand robustness by attacking models using **FGSM (Fast Gradient Sign Method)** and **PGD (Projected Gradient Descent)**, and defending against them using **Adversarial Training**. Dataset: **CIFAR-10**.

### 4.1 Implementation of Attacks (30 Points)

#### 4.1.1 Setup

* Train a **Simple CNN** (2-3 Convolutional layers) on CIFAR-10.
* Train for at least 10 epochs.
* Report accuracy on clean data.

#### 4.1.2 Implementation of FGSM & PGD

* **FGSM:** Implement the attack.
* **PGD:** Implement PGD.
* Step size .
* Number of iterations .


* **Epsilon ():** Test with .

#### 4.1.3 Evaluation under Attack

* Calculate the accuracy of the Simple CNN under:
1. Clean Data.
2. FGSM Attack.
3. PGD Attack ( and ).


* **Plots:** Plot **Accuracy vs. Epsilon** (4 curves: Clean, FGSM, PGD-5, PGD-10).

#### 4.1.4 Analysis

* Which attack is stronger? Why?
* How does increasing  affect accuracy?
* Why does PGD usually degrade performance more than FGSM?

### 4.2 Architecture Comparison (30 Points)

#### 4.2.1 ResNet18

* Load a **ResNet18** model (pretrained or trained from scratch on CIFAR-10).
* Repeat the attacks (FGSM, PGD) on ResNet18.

#### 4.2.2 Comparison

* Compare the robustness of **Simple CNN** vs. **ResNet18**.
* **Analysis:** Does a deeper network (ResNet) provide more robustness? Discuss the role of "Skip Connections" and the smoothness of the decision boundary.

### 4.3 Adversarial Training (40 Points)

**Goal:** Improve robustness by training on attacked images.

#### 4.3.1 Implementation

* **Method:** Instead of training on just clean images, generate PGD adversarial examples ( or similar) *during* the training loop.
* **Train:** Train the ResNet18 (or Simple CNN) using these adversarial examples (or a mix of clean + adversarial).
* Train for 10 epochs.

#### 4.3.2 Evaluation (Defense)

* Evaluate the **Adversarially Trained Model** against:
1. Clean Data.
2. FGSM Attack.
3. PGD Attack.


* **Comparison:** Compare the results with the "Standard Trained" model from Section 4.2.

#### 4.3.3 Plots & Conclusion

* Plot **Accuracy vs. Epsilon** for the Adversarially Trained model.
* **Final Plot:** Overlay the curves for Standard Training vs. Adversarial Training.
* **Analysis:**
* Did Adversarial Training improve robustness against attacks?
* Did the accuracy on **Clean Data** drop? (This is a common trade-off; discuss it).
* Is the model now robust against FGSM even though it was trained on PGD?