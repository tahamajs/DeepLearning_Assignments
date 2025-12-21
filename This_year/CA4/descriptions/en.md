

# University of Tehran

**Faculty of Electrical & Computer Engineering**
**Course:** Deep Learning and Neural Networks
**Assignment:** Homework 4

**Designers:** Ali Ramzani, Ehsan Foroutan
**Deadline:** December 21, 2025 (Azar 30, 1404)

---

## Question 1: RNN Models for Joint Tasks in Spoken Language Understanding (SLU)

**Context:**
In spoken dialogue systems, Spoken Language Understanding (SLU) is a critical component typically involving two sub-tasks: **Intent Detection** and **Slot Filling**. In this assignment, we will implement neural models based on RNNs to address these two tasks jointly.

**Task Description:**

* **Intent Detection:** Determine the user's intention (e.g., `atis_flight`, `atis_airfare`). This is a sequence classification task.
* **Slot Filling:** Extract semantic constituents from the utterance (e.g., origin, destination, date). This is a sequence labeling task using **BIO** tagging (Begin, Inside, Outside).
* Example: "Show me flights from Boston to Denver"
* Slots: `O O O O B-fromloc.city_name O B-toloc.city_name`
* Intent: `atis_flight`



**Dataset:**
We will use the **ATIS** (Airline Travel Information System) dataset.
Code to download the dataset (from Kaggle):

```python
import kagglehub
path = kagglehub.dataset_download("siddhadev/atis-dataset-clean")
print("Path to dataset files:", path)

```

### 1.1 Data Preparation (12 Points)

1. **Exploration:** Load the dataset and check the number of training and test samples, the number of intents, and the number of slot labels. Separate the validation set.
2. **Tokenization:**
* First, tokenize the data. Since the ATIS dataset is relatively clean, **white space tokenization** is sufficient.
* Build three vocabularies: `Word Vocabulary`, `Slot Vocabulary`, and `Intent Vocabulary`.


3. **Special Tokens:** Be careful to include `<PAD>` and `<UNK>` tokens in your vocabularies. Explain when each is used.
4. **Collate Function:**
* Implement a `collate_fn` for your DataLoader.
* This function should pad sentences in a batch to the length of the longest sentence in that specific batch (dynamic padding).
* Compare the length of the padded sentences in the first and second batch. What is the benefit of batch-level padding vs. dataset-level padding?



### 1.2 Baseline Model: BiRNN (16 Points)

Implement a Bi-directional RNN model for **Slot Filling only**.

* **Architecture (Table 1):** Embedding -> RNN -> Dropout -> Linear.
* **Hyperparameters (Table 2):** Embed Dim: 128, Hidden Dim: 128, Bidirectional: True, Dropout: 0.5, Batch Size: 32, Epochs: 10, LR: 0.001, Optimizer: Adam.
* **Evaluation:**
* Train the model and plot the Loss and Accuracy curves.
* Report **F1-score** (using `seqeval` library) and the **Classification Report** on the test set.
* **Note:** Specifically for Slot Filling, accuracy is not enough (due to the high number of 'O' tags). You must report F1.



### 1.3 BiLSTM Joint Model (24 Points)

Implement a single network that performs both Intent Detection and Slot Filling jointly.

* **Architecture (Table 3):**
* Embedding -> LSTM -> Dropout
* **Head 1 (Intent):** Linear layer (takes the final hidden state or pooled output).
* **Head 2 (Slot):** Linear layer (takes sequence of hidden states).


* **Training:** Define the loss function as the sum of the Intent Loss and Slot Loss.
* **Evaluation:**
* Train the model. Plot Loss and Accuracy for both tasks.
* Report Accuracy for Intent Detection.
* Report F1-score and Classification Report for Slot Filling.
* Does solving these tasks jointly improve performance compared to separate models?



### 1.4 Encoder-Decoder Non-aligned Joint Model (48 Points)

Implement an **Encoder-Decoder** architecture.

* **Concept:** Unlike the BiLSTM where input and output lengths are aligned one-to-one, an Encoder-Decoder allows for more complex relationships.
* **Special Tokens:** You must handle `<BOS>` (Begin of Sentence) and `<EOS>` (End of Sentence) tokens manually for the decoder generation process.
* **Architecture:**
* **Encoder (Table 4):** Embedding (128) -> LSTM (128).
* **Decoder (Table 5):** Embedding (64) -> LSTM (256) -> Dropout -> Intent Head (Linear) & Slot Head (Linear).


* **Hyperparameters (Table 6):** As listed above.
* **Implementation:**
* The Decoder generates the sequence. At each step, it predicts the slot.
* The Intent is usually predicted at the end (or from the Encoder's context).


* **Evaluation:**
* Train the model.
* Report Accuracy for Intent and F1-score for Slot Filling (using `seqeval`).
* **Analysis:** Compare the Slot Filling performance of this Encoder-Decoder model against the BiLSTM model.
* **Question:** Why might a standard Encoder-Decoder (without CRF or Beam Search, using greedy decoding) perform worse on Slot Filling than a direct BiLSTM? Discuss the issue of "greedy" decoding in this context.



---

## Question 2: Stock Price Prediction using Recursive Architectures

**Goal:** Predict the **S&P 500** stock price using Deep Learning.
**Library:** `yfinance`

```python
import yfinance as yf
df = yf.download("^GSPC", start="2000-01-01", auto_adjust=False)

```

### 2.1 Data Preprocessing (20 Points)

1. **Feature Engineering:**
* Load data (Open, High, Low, Close).
* Extract date features: Year, Month, Day.
* **Cyclical Encoding:** Apply Sine and Cosine transformations to Month and Day features to preserve their cyclic nature. Explain why this is necessary.
* **MinMax Scaling:** Apply MinMax scaling to the 'Year' feature.


2. **Stationarity:**
* Plot the **ACF** (Autocorrelation Function) and **PACF** (Partial Autocorrelation Function) for one feature (e.g., Close Price) using `statsmodels`.
* Analyze how these plots help determine the input window size and prediction horizon.
* **Differencing:** Apply differencing to the signal. Re-plot ACF/PACF. Analyze the effect.
* **Research:** Explain "Stationarizing" and the **Augmented Dickey–Fuller (ADF)** test. Is the signal stationary?


3. **Ergodicity:** Research the concept of Ergodicity. Are stock prices Ergodic?
4. **Windowing & Splitting:**
* Based on the ACF/PACF analysis, determine the Input Window size and Output Horizon.
* **Data Splitting:** Train (70%), Validation (20%), Test (10%). **Crucial:** You must split by time (chronologically) to avoid **Data Leakage**. Do not shuffle!
* Test set should cover the period from the beginning of 2024 to the end of 2025.


* **Normalization:** Apply **Z-score** normalization (fit on Train, transform on Val/Test). Plot histograms to verify.



### 2.2 Baseline Model (10 Points)

To evaluate the complexity and performance of advanced models, we need a baseline (often referred to as a NARMAX network in literature).

1. **Architecture:** Implement an **MLP** (Multi-Layer Perceptron) with 2 hidden layers, **Batch Normalization**, and **Dropout**.
2. **Training:**
* Optimizer: AdamW (with Weight Decay).
* Loss: MSE.
* Save the model with the lowest Validation Loss.


3. **Evaluation:**
* Report RMSE, MAE, MAPE, and R2 scores on the Test set.
* Plot the predictions vs. actual prices for the 2024-2025 period.
* Validation Loss should be less than 0.70 (normalized scale).



### 2.3 Recursive Neural Network Models (50 Points)

Recursive architectures are ideal for time-series forecasting as they filter noise and maintain state. We will implement **Non-Linear State-Space Based Networks**.

#### 2.3.1 LSTM Architecture

* **Structure:** **CNN + LSTM**.
* Layer 1: **1D Convolution**. Explain its purpose (feature extraction from local windows).
* Layer 2: **LSTM**.


* **Details:**
* Should the LSTM receive the last output of the CNN or the full sequence? Explain.
* Should you use the last hidden state of the LSTM or the full sequence for the final prediction layer?


* **Training:** Train the model. Validation loss must be < 0.02.
* **Reporting:** Report hyperparameters, architecture, and parameter count.

#### 2.3.2 GRU Architecture

* **Structure:** **CNN + GRU**.
* **Comparison:** Replace the LSTM layer with a **GRU** layer. Keep the network depth and parameters as similar as possible to the LSTM model for a fair comparison.
* **Training:** Train the model. Validation loss must be < 0.02.

### 2.4 Comparison and Analysis (20 Points)

1. **Metrics:** Create a comparison table reporting RMSE, MAE, MAPE, and R2 for all models (Baseline, LSTM, GRU) on the Test set.
2. **Efficiency:** Report the number of parameters, FLOPs (approximate), and Training Time for each model. Mention the Hardware used (GPU/CPU, Colab/Kaggle/Local, e.g., MPS on Mac or CUDA).
3. **Visualization:**
* Plot the predictions of all models vs. Actual Prices for the 2024-2025 range on a single graph.
* **Long-term Prediction:** Analyze how these models perform for long-term forecasting (e.g., 1 year ahead). Since the model predicts the next step, you must feed the prediction back as input recursively. Discuss the stability and error propagation in this scenario (2026-2027).



### 2.5 Bonus: CNN-Only Network (5 Points)

* **Task:** Design a network consisting **only of Convolutional Layers** (e.g., TCN - Temporal Convolutional Network style or Dilated Convolutions).
* **Constraint:** It should be lighter (fewer parameters) than the Baseline.
* **Analysis:** Can a CNN-only network capture temporal dependencies effectively?
* **Deliverables:** Train the model, plot Loss curves, and plot predictions for 2024-2025. Compare results with previous models.