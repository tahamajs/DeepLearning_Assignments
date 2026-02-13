# CA4 - Time Series Prediction

This project forecasts physiological signals using both statistical and deep sequence models.

## Notebook
- `code/NNDL_CA4_2_1.ipynb`

## Data Pipeline
- Reads train/validation/test CSV files.
- Grouping by patient/sequence ID for per-subject temporal modeling.
- Feature scaling with `MinMaxScaler`.
- Sliding-window dataset creation for supervised forecasting.

## Implemented Models

### 1) Statistical baselines
- `SARIMAX` configurations with exogenous signal (`O2Sat`) for `HR` forecasting.

### 2) Neural models
- `MarkovPredictor` (dense transition-style model on windowed inputs).
- `GRUPredictor`.
- `LSTMPredictor`.
- Bidirectional GRU and bidirectional LSTM variants.

## Training
- Loss: MSE.
- Optimizer: SGD.
- Optional early stopping during train/val runs.
- Additional train+val combined training for final model comparisons.

## Forecasting Modes
- One-step supervised prediction.
- Multi-step recursive forecasting via `predict_multi_steps`.

## Evaluation Metrics
- MSE, RMSE, MAE, MAPE, R2.
- Explained variance and correlation-based diagnostics.
- Comparative plots across SARIMAX and neural models.

## Methods Used (Checklist)
- Classical time-series modeling (SARIMAX).
- Window-based deep forecasting.
- Recurrent and bidirectional recurrent architectures.
- Multi-step autoregressive rollout evaluation.

## Files
- `code/NNDL_CA4_2_1.ipynb`
- `report/NNDL_UT_CA4_Q2.pdf`
- `description/`
- `paper/`

## Run
1. Open notebook.
2. Confirm CSV paths in `dataset/`.
3. Run preprocessing -> model training -> evaluation blocks.
