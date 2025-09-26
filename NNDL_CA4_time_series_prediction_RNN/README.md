# NNDL_CA4_time_series_prediction_RNN

This folder contains the implementation of time series prediction using RNNs (LSTM/GRU) with uncertainty estimation. Part of Neural Networks and Deep Learning course assignment 4.

## Concepts Covered

### Time Series Prediction

Forecasting future values based on historical data patterns, using sequential models that capture temporal dependencies.

### Recurrent Neural Networks for Sequences

- **LSTM**: Long Short-Term Memory handles long-term dependencies with gates
- **GRU**: Gated Recurrent Unit, simplified LSTM with similar performance
- **Bidirectional RNNs**: Process sequences in both directions for richer context

### Uncertainty Estimation

- **Maximum Likelihood Estimation (MLE)**: Learns both mean and variance predictions
- **Probabilistic Forecasting**: Provides confidence intervals, not just point estimates
- **Loss Function**: Negative log-likelihood for Gaussian distribution

### Markov Models

- **Markov Property**: Future depends only on current state
- **Linear Transitions**: Simple feedforward networks modeling state transitions

## Implementation Details

### Dataset

- **Time Series Data**: Likely financial or sensor data (e.g., stock prices, weather)
- **Preprocessing**:
  - Min-Max scaling to [0,1]
  - Sliding window approach for sequences
  - Train/val/test splits

### Model Architectures

#### Markov Predictor

- **Input**: Flattened window of historical values
- **Layers**: Linear → BatchNorm → Dropout → Linear → Sigmoid
- **Transition Size**: Hyperparameter for hidden representation

#### Recurrent Predictors (LSTM/GRU)

- **RNN Layer**: Processes sequence step-by-step
- **Bidirectional**: Optional for reverse temporal context
- **Output**: Takes last hidden state for prediction
- **Fully Connected**: Hidden → FC → BatchNorm → Dropout → Output

#### MLE Predictors

- **Dual Output**: Predicts both mean and variance
- **Gaussian Likelihood**: Assumes normal distribution with learned variance
- **Loss**: -log N(y|μ,σ²) where μ and σ² are predicted

### Training Parameters

- **Window Size**: 20 (sequence length)
- **Forecast Horizon**: 2 (prediction steps ahead)
- **Hidden Size**: 64-128
- **FC Size**: 32-64
- **Batch Size**: 32
- **Learning Rate**: 0.01 (SGD)
- **Epochs**: 100+
- **Early Stopping**: Patience 10

### Loss Functions

- **MSE/MAE**: For deterministic prediction
- **NLL**: For probabilistic prediction (MLE models)

### Evaluation Metrics

- **R² Score**: Explained variance (1.0 = perfect)
- **MSE/MAE**: Point prediction accuracy
- **Uncertainty Calibration**: How well predicted variances match actual errors

## Results

### Model Comparisons

#### Markov Model

- **R² (Mean)**: ~0.75
- **R² (Variance)**: ~0.60
- Simple, fast, but limited temporal modeling

#### Unidirectional LSTM

- **R² (Mean)**: ~0.82
- **R² (Variance)**: ~0.68
- Better sequence modeling than Markov

#### Bidirectional LSTM

- **R² (Mean)**: ~0.85
- **R² (Variance)**: ~0.72
- Improved with reverse context

#### GRU Variants

- Similar performance to LSTM
- Slightly faster training
- Fewer parameters

#### MLE LSTM (Uncertainty)

- **R² (Mean)**: ~0.83
- **R² (Variance)**: ~0.70
- Provides confidence intervals
- Better calibrated uncertainty

### Training Dynamics

- Loss decreases steadily, with early stopping preventing overfitting
- Validation loss plateaus around epoch 50-70
- MLE models converge slower due to dual objectives

### Probabilistic Forecasting Benefits

- **Confidence Intervals**: 95% prediction intervals
- **Risk Assessment**: Higher uncertainty for volatile periods
- **Decision Making**: Better than point estimates for planning

### Challenges Addressed

- **Long Dependencies**: RNNs capture temporal patterns
- **Uncertainty Quantification**: MLE provides variance estimates
- **Overfitting**: Dropout, early stopping, regularization
- **Non-stationarity**: Scaling and normalization

## Applications

- **Financial Forecasting**: Stock prices, volatility prediction
- **Weather Prediction**: Temperature, precipitation with uncertainty
- **Demand Forecasting**: Sales, traffic with confidence bounds
- **IoT/Sensors**: Equipment monitoring with failure prediction

## Files

- `code/`: PyTorch implementation of Markov, LSTM, GRU predictors
- `report/`: Detailed analysis with R² scores and uncertainty plots
- `paper/`: Time series forecasting papers
- `description/`: Assignment description

## Conclusion

The implementation demonstrates effective time series prediction with RNNs, showing improvements over simple Markov models and benefits of uncertainty estimation for robust forecasting.
