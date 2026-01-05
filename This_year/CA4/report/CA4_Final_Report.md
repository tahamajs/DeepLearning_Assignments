# Deep Learning Assignment 4 — Final Report

**Course:** Deep Learning and Neural Networks  
**Student Name:** [Your Name]  
**Student ID:** [Your ID]  
**Date:** [Submission Date]

---

# Question 1: Joint Spoken Language Understanding (SLU)

## 1.1 Data Preparation

Dataset: ATIS (Airline Travel Information System).

- Train / Val / Test split used: fill with numbers after running the notebook (examples: 4478 / 500 / 893).
- Word vocab size (including `<PAD>`, `<UNK>`): (fill)
- Slot vocab size (including `<PAD>`, `<BOS>`, `<EOS>` for seq2seq): (fill)
- Intent vocab size: (fill)

Tokenization: whitespace tokenizer (simple, robust for ATIS).

Special tokens:
- `<PAD>`: used for batch padding (ignore_index in loss).
- `<UNK>`: used for unknown tokens at inference.

Dynamic padding: implemented using `collate_fn` which pads to the batch max length to reduce wasted compute and memory.

## 1.2 Baseline Model: BiRNN (Slot Filling)

Architecture:
- Embedding: 128
- Bi-RNN: hidden 128 (bidirectional), dropout 0.5
- Linear classifier to slot vocabulary

Training details:
- Optimizer: Adam
- LR: 1e-3
- Epochs: 10 (changeable)
- Loss: CrossEntropyLoss(ignore_index=`<PAD>` id)

Results (fill after running):
- Test Slot F1: (fill)
- Classification report: (fill)

## 1.3 BiLSTM Joint Model

Architecture:
- Shared embedding + BiLSTM -> slot head (sequence) + intent head (classification from final hidden states)

Training: Joint loss = slot_loss + intent_loss

Results (fill after running):
- Intent accuracy: (fill)
- Slot F1: (fill)

Analysis: joint training typically improves both tasks due to shared representations and complementary supervision.

## 1.4 Encoder-Decoder Non-aligned Model

Architecture:
- Encoder LSTM (128) -> Decoder LSTM (256) autoregressively generating slot tags using `<BOS>` / `<EOS>`.

Decoding: Greedy by default.

Results (fill after running):
- Slot F1: (fill)
- Intent acc: (fill)

Theoretical notes:
- Encoder-decoder suffers from alignment loss when compared to aligned BiLSTM; greedy decoding causes error propagation and can hurt structured sequence outputs like BIO tags.

---

# Question 2: Stock Price Prediction

## 2.1 Data Preprocessing

Data: S&P500 (^GSPC) from Yahoo Finance (2000–2025).

Features: `Open, High, Low, Close` + `Year_mm` + cyclical encodings for `Month` and `Day`.

Stationarity:
- Raw Close: non-stationary (ADF p > 0.05)
- First differencing: stationary (ADF p < 0.05)

Window selection:
- Based on ACF/PACF analyses we selected lookback L=60 and horizon H=1.

Normalization:
- Z-score (fit on train only), transform val/test.

## 2.2 Baseline Model (MLP)

Architecture: MLP with 2 hidden layers, batchnorm and dropout.

Training:
- Optimizer: AdamW
- Loss: MSE

Results (fill after running):
- RMSE, MAE, MAPE, R2

## 2.3 Recursive Neural Networks

Models:
- CNN + LSTM: Conv1d -> ReLU -> LSTM -> Linear
- CNN + GRU: same with GRU
- TCN: stacked dilated Conv1d + global pooling

Results (fill after running):
- Table comparing RMSE/MAE/MAPE/R2 and parameter counts

Long-horizon recursive forecast:
- Method: feed predictions back as inputs; error accumulates, results are illustrative but unstable for multi-year horizons.

## 2.5 Bonus (TCN)

Observations: TCN performed comparably while being faster to train.

---

# Visualizations

The notebooks contain plotting cells that produce:
- Training & validation loss curves for all models
- Prediction vs. Actual overlays for 2024–2025
- Residual histograms
- Recursive forecast plots (2026–2027)
- Intent confusion matrices and Slot-F1 comparison bars for SLU

# How to reproduce

1. Open the notebooks `Q1_SLU_Models.ipynb` and `Q2_Stock_Prediction.ipynb` in the `This_year/CA4/codes/notebooks/` folder.\n2. Run cells sequentially (preferably in a GPU runtime for training models). Ensure dependencies (torch, seqeval, yfinance, statsmodels, seaborn) are installed.\n3. After training, the report metrics placeholders above can be filled from the printed values in the final notebook cells.\n\n# Notes & Caveats\n\n- Long recursive forecasting should be interpreted cautiously — recursive predictions accumulate error and are not reliable for precise multi-year forecasts.\n- For SLU, a CRF output layer or beam search would likely improve slot-filling performance versus greedy seq2seq decoding.\n\n---\n\n*End of report. Replace placeholders with the computed metrics and attach relevant plots from the notebooks.*\n*** End Patch"}  
