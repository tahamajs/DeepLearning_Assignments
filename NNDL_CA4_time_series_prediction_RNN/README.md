# NNDL_CA4_time_series_prediction_RNN

This folder contains the implementation of RNN-based models for time series prediction. Part of Neural Networks and Deep Learning course assignment 4.

## Concepts Covered

### Time Series Prediction

Forecasting future values based on historical data sequences.

### Recurrent Neural Networks (RNNs)

- **Basic RNN**: Processes sequences step-by-step, maintaining hidden state
- **LSTM/GRU**: Advanced RNN variants that handle long-term dependencies
- **Bidirectional RNN**: Considers both past and future context

### Sequence Modeling

- Input: Time series data (univariate or multivariate)
- Output: Predicted values for future time steps
- Windowing: Sliding windows for supervised learning

### Training

- Loss: Mean Squared Error (MSE) for regression
- Optimization: Adam, RMSProp
- Handling sequences: Padding, masking for variable lengths

### Evaluation

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Visualization: Predicted vs. actual plots

### Challenges

- Long-term dependencies
- Overfitting on noisy data
- Computational complexity for long sequences

### Applications

- Stock price prediction
- Weather forecasting
- Demand forecasting
- Anomaly detection

## Files

- `code/`: RNN/LSTM/GRU implementations
- Dataset: Time series data (e.g., stock prices, weather)

## Results

RNN models capture temporal patterns, with LSTM/GRU outperforming basic RNNs on long sequences.
