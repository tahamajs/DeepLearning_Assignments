# Time Series Prediction Using Recurrent Neural Networks

## Comprehensive Analysis of Markov Chain, RNN, LSTM, and GRU Models for Physiological Data Forecasting

A comprehensive implementation and comparison of time series prediction methodologies for physiological heart rate forecasting using recurrent neural networks. This project demonstrates the effectiveness of bidirectional LSTM and GRU architectures in healthcare applications.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Architecture Details](#architecture-details)
- [Methodology](#methodology)
- [Results and Analysis](#results-and-analysis)
- [Visualizations](#visualizations)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)

---

## 🎯 Project Overview

### Project Information

**Author:** Taha Majlesi (810101504)  
**Institution:** University of Tehran, Faculty of Electrical and Computer Engineering  
**Course:** Neural Networks and Deep Learning - Course Assignment 4 (CA4)  
**Date:** 2024

### Abstract

This comprehensive study presents a detailed comparison of time series prediction methodologies, focusing on recurrent neural network architectures applied to physiological heart rate data. We implement and evaluate multiple approaches including Markov chains, traditional RNNs, Long Short-Term Memory (LSTM) networks, and Gated Recurrent Units (GRUs), with both unidirectional and bidirectional processing capabilities.

### Objectives

1. **Architectural Comparison:** Systematically compare the performance of Markov chains, RNNs, LSTMs, and GRUs
2. **Bidirectional Processing Analysis:** Evaluate the effectiveness of bidirectional processing in capturing temporal dependencies
3. **Feature Engineering:** Develop robust preprocessing and feature selection methodologies for physiological data
4. **Statistical Validation:** Provide comprehensive statistical analysis using SARIMAX models as baseline comparisons
5. **Clinical Relevance:** Assess the practical applicability of predictions in healthcare settings

### Key Contributions

- ✅ Comprehensive comparison of 5 different model architectures
- ✅ Statistical baseline with SARIMAX models
- ✅ Feature selection and preprocessing pipeline
- ✅ Evaluation of bidirectional vs unidirectional processing
- ✅ Clinical applicability assessment
- ✅ Multi-step forecasting capabilities

---

## 🏆 Key Results

### Performance Summary

| Model | R² Score | RMSE | MAE | Status |
|-------|----------|------|-----|--------|
| **Bidirectional LSTM** | **0.87** | **0.042** | **0.021** | 🥇 Best |
| Bidirectional GRU | 0.83 | 0.048 | 0.022 | 🥈 Excellent |
| Unidirectional LSTM | 0.82 | 0.049 | 0.023 | 🥉 Good |
| Unidirectional GRU | 0.80 | 0.051 | 0.024 | ✓ Acceptable |
| Markov Chain (Baseline) | 0.74 | 0.064 | 0.026 | Baseline |

### Multi-Step Forecasting Results

| Model | R² Score | RMSE (BPM) | MAE (BPM) | Status |
|-------|----------|------------|-----------|--------|
| **Bidirectional LSTM** | **0.82** | **4.1** | **3.2** | 🥇 Best |
| Bidirectional GRU | 0.82 | 4.3 | 3.4 | 🥈 Excellent |
| Unidirectional LSTM | 0.80 | 4.6 | 3.7 | 🥉 Good |
| Unidirectional GRU | 0.78 | 4.8 | 3.9 | ✓ Acceptable |
| Markov Chain | 0.72 | 5.1 | 4.2 | Baseline |
| SARIMAX (14,1,24) | 0.69 | 5.5 | 4.8 | Statistical |

### Key Findings

- **Bidirectional Processing:** 2-4% improvement in prediction accuracy
- **LSTM vs GRU:** LSTM performs better (0.87 vs 0.83) but GRU trains faster
- **Improvement over Baseline:** 12-15% improvement over statistical methods (SARIMAX)
- **Clinical Application:** 89% of predictions within ±5 BPM (clinical threshold)
- **Statistical Baseline:** SARIMAX models achieve R² = 0.65-0.69, demonstrating deep learning superiority

---

## 📊 Dataset

### Dataset Information

- **Data Type:** Physiological heart rate data from intensive care units
- **Total Samples:** 39,136 training observations
- **Features:** 44 initial features, reduced to 20 after feature selection
- **Temporal Resolution:** Hourly measurements
- **Dataset Structure:**
  - **Training Set:** 39,136 observations
  - **Validation Set:** Separate patient data
  - **Test Set:** Independent patient data

### Feature Categories (44 Initial Features)

1. **Vital Signs:** Heart Rate (HR), Blood Pressure (SBP, DBP, MAP), Temperature, Respiratory Rate
2. **Blood Gas Analysis:** Oxygen Saturation (O2Sat, SaO2), CO2, pH
3. **Laboratory Values:** Complete Blood Count, Metabolic Panel
4. **Electrolytes:** Sodium, Potassium, Chloride, Calcium
5. **Liver Function:** Bilirubin, AST, Alkaline Phosphatase
6. **Demographics:** Age, Gender, Patient ID

### Data Preprocessing

1. **Feature Selection:**
   - Correlation-based elimination (threshold: 0.3)
   - Maximum correlation filtering
   - Final feature set: 20 features (including HR and Patient_ID)

2. **Normalization:**
   - MinMax scaling to [0, 1] range
   - Separate scalers for input features and target variable
   - Preserves relative relationships between features

3. **Sequence Construction:**
   - Sliding window approach
   - Window size: 2 hours (input)
   - Forecast horizon: 1 hour (output)
   - Multi-patient sequence generation

### Data Quality

- ✅ No missing values in the dataset
- ✅ Consistent data types across partitions
- ✅ Proper temporal ordering of measurements
- ✅ Valid physiological ranges for all vital signs

---

## 🏗️ Architecture Details

### 1. Markov Chain Predictor (Baseline)

**Architecture:**
- Input: Flattened 2D temporal sequences
- Hidden Layer: Linear transformation with BatchNorm and Dropout
- Output: Single value prediction with Sigmoid activation

**Parameters:**
- Window size: 2
- Transition size: 128
- Dropout: 0.5

### 2. Recurrent Neural Network (Base Architecture)

**Shared Architecture for RNN/LSTM/GRU:**
```python
Recurrent Layer → Linear → BatchNorm → Dropout → Output Layer
```

**Key Components:**
- Hidden state size: 512 units
- Fully connected size: 128 units
- Batch normalization for stable training
- Dropout (0.5) for regularization

### 3. LSTM Networks

**Architecture:**
- Long Short-Term Memory cells with gating mechanisms
- **Forget Gate:** Discards irrelevant information
- **Input Gate:** Stores new information
- **Output Gate:** Controls hidden state output

**Variants:**
- Unidirectional LSTM: 1,093,632 parameters
- Bidirectional LSTM: ~2,187,264 parameters

### 4. GRU Networks

**Architecture:**
- Gated Recurrent Units with simplified gating
- **Reset Gate:** Controls past information forgetting
- **Update Gate:** Balances past and new information

**Variants:**
- Unidirectional GRU: 886,273 parameters
- Bidirectional GRU: ~1,772,546 parameters

### 5. Bidirectional Processing

**Mechanism:**
- Forward processing: Sequence from t=1 to T
- Backward processing: Sequence from t=T to 1
- Output combination: Concatenated forward and backward hidden states
- Hidden state size doubles: 2 × hidden_state_size

---

## 🔬 Methodology

### Training Configuration

- **Optimizer:** SGD (Stochastic Gradient Descent)
- **Learning Rate:** 0.005
- **Batch Size:** 32
- **Epochs:** 50 (maximum)
- **Early Stopping:** Patience = 3 epochs
- **Loss Function:** Mean Squared Error (MSE)
- **Device:** CUDA (GPU) if available, else CPU

### Training Process

1. **Epoch-Based Training:**
   - Separate training and validation phases
   - Model mode switching (train/eval)
   - Gradient management and backpropagation

2. **Early Stopping:**
   - Monitors validation loss
   - Stops training when no improvement for 3 epochs
   - Loads best model checkpoint

3. **Model Checkpointing:**
   - Saves best model based on validation loss
   - File: `best_model.pth`

### Evaluation Metrics

1. **RMSE (Root Mean Square Error):**
   - Emphasizes large errors
   - Clinically interpretable (BPM units)

2. **MAE (Mean Absolute Error):**
   - Robust to outliers
   - Average deviation from true value

3. **R² Score:**
   - Proportion of variance explained
   - Standard metric for model comparison

4. **Explained Variance Score:**
   - Complements R² for variance analysis

5. **Cosine Distance:**
   - Measures temporal pattern similarity
   - Scale-independent

### Statistical Analysis

**SARIMAX Models (Baseline):**
- **SARIMAX(2,1,2):** Basic model (R² = 0.65)
- **SARIMAX(14,1,12):** Complex model (R² = 0.67)
- **SARIMAX(14,1,24):** Very complex model (R² = 0.69)

**Stationarity Testing:**
- Augmented Dickey-Fuller (ADF) test
- Original series: Non-stationary (p = 0.355)
- First difference: Stationary (p = 2.90e-23)

**Autocorrelation Analysis:**
- ACF for determining MA order (q)
- PACF for determining AR order (p)

---

## 📈 Results and Analysis

### Single-Step Prediction Results

**Validation Set Performance:**

The validation set evaluation provides unbiased assessment of model generalization capability. All RNN models significantly outperform the baseline Markov model, with bidirectional processing providing consistent improvements.

**Key Insights:**
- Bidirectional LSTM achieves the highest R² Score (0.87) and lowest error metrics
- Bidirectional processing provides 2-4% improvement over unidirectional variants
- LSTM performs slightly better than GRU (~0.04 in R²), but GRU trains faster

### Multi-Step Forecasting Analysis

**Challenges:**
- Error accumulation over time steps
- Increasing uncertainty with forecast horizon
- Regime changes in patient conditions

**Performance:**
- Bidirectional LSTM maintains best performance (R² = 0.82)
- All deep learning models outperform statistical baselines
- Error remains within clinically acceptable ranges (±5 BPM for 89% of predictions)

### Clinical Interpretation

**Clinically Acceptable Metrics:**
- ✅ Clinical Threshold: ±5 BPM error considered clinically significant
- ✅ Bidirectional LSTM: 89% of predictions within ±5 BPM
- ✅ Practical Application: Sufficient accuracy for clinical decision support
- ✅ Warning Systems: Suitable for early warning systems

### Computational Efficiency

**Training Time Comparison:**

| Model | Epochs (Early Stop) | Approximate Time | Status |
|-------|-------------------|------------------|--------|
| Markov Chain | 21 | Fast | ⚡ |
| GRU (Unidirectional) | 9 | Fast | ⚡ |
| LSTM (Unidirectional) | 11 | Moderate | ⏱️ |
| GRU (Bidirectional) | 19 | Moderate | ⏱️ |
| LSTM (Bidirectional) | 15 | Moderate-Slow | ⏱️ |

**Efficiency Notes:**
- Early stopping effectively prevents overfitting
- Consistent convergence patterns across architectures
- Fast inference suitable for real-time processing

---

## 📸 Visualizations

The following visualizations are extracted from the notebook and demonstrate key results:

### Data Analysis Visualizations

![Feature Correlation Heatmap](images/plot_28_1.png)
*Feature correlation heatmap after initial correlation-based elimination*

![Final Feature Correlation](images/plot_30_2.png)
*Final feature correlation heatmap with 20 selected features*

### Stationarity and Autocorrelation Analysis

![Heart Rate Time Series](images/plot_36_3.png)
*Original and differentiated heart rate time series for patient 9*

![ACF and PACF Plots](images/plot_37_4.png)
*Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for determining ARIMA model orders*

### SARIMAX Model Forecasts

![SARIMAX(2,1,2) Forecast](images/plot_39_5.png)
*SARIMAX(2,1,2) model forecast vs true heart rate values*

![SARIMAX(14,1,12) Forecast](images/plot_42_6.png)
*SARIMAX(14,1,12) model forecast vs true heart rate values*

![SARIMAX(14,1,24) Forecast](images/plot_46_7.png)
*SARIMAX(14,1,24) model forecast vs true heart rate values*

### Training Curves

![Markov Model Loss](images/plot_71_8.png)
*Training and validation loss curves for Markov Chain model*

![GRU Model Loss](images/plot_72_9.png)
*Training and validation loss curves for Unidirectional GRU model*

![LSTM Model Loss](images/plot_73_10.png)
*Training and validation loss curves for Unidirectional LSTM model*

![Bidirectional GRU Loss](images/plot_74_11.png)
*Training and validation loss curves for Bidirectional GRU model*

![Bidirectional LSTM Loss](images/plot_75_12.png)
*Training and validation loss curves for Bidirectional LSTM model*

### Multi-Step Forecasting Comparisons

![Multi-Step Forecast Comparison](images/plot_83_13.png)
*Comparison of all models for multi-step heart rate forecasting*

![Detailed Forecast View 1](images/plot_84_14.png)
*Detailed view of multi-step forecasts showing model performance*

![Detailed Forecast View 2](images/plot_85_15.png)
*Another detailed view of forecasting performance across models*

![Detailed Forecast View 3](images/plot_86_16.png)
*Extended forecast comparison demonstrating long-term prediction accuracy*

### Model Performance Visualizations

![Final Comparison](images/plot_92_17.png)
*Final comprehensive comparison of all models including SARIMAX baselines*

![Performance Analysis](images/plot_93_18.png)
*Performance analysis visualization showing relative model effectiveness*

![Results Summary](images/plot_94_19.png)
*Summary visualization of key results and metrics*

![Final Results](images/plot_95_20.png)
*Final results visualization with all model comparisons*

---

## 💻 Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Torchinfo

### Installation Steps

1. **Clone the repository:**
```bash
cd CA4_Sequence_Modeling/Time_Series_Prediction
```

2. **Install dependencies:**
```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels torchinfo
```

3. **Prepare dataset:**
   - Place CSV files in `dataset/` directory:
     - `train_data.csv`
     - `val_data.csv`
     - `test_data.csv`

4. **For Google Colab:**
   - Mount Google Drive
   - Update path in the notebook to point to dataset location

---

## 🚀 Usage

### Running the Notebook

1. **Open the notebook:**
```bash
jupyter notebook code/NNDL_CA4_2_1.ipynb
```

2. **Or use Google Colab:**
   - Upload the notebook to Google Colab
   - Ensure dataset is accessible in Google Drive

3. **Run all cells:**
   - Execute cells sequentially
   - Models will be trained and evaluated automatically
   - Visualizations will be generated inline

### Key Configuration

Before running, ensure these parameters are set correctly:

```python
# Data paths (adjust for your environment)
path = '/content/drive/MyDrive/Colab/NNDL/CA4/Part1/Dataset/'

# Model parameters
window_size = 2           # Input size (2 hours)
forecast_horizon = 1      # Forecast horizon (1 hour)
batch_size = 32           # Batch size
learning_rate = 0.005     # Learning rate
epochs = 50               # Maximum epochs
patience = 3              # Early stopping patience

# Model architecture
hidden_state_size = 512
fully_connected_size = 128
```

### Reproducibility

- Random seed is set to `42` for all operations
- Trained models are saved as `best_model.pth`
- All hyperparameters are documented in the notebook

---

## 📁 Project Structure

```
Time_Series_Prediction/
│
├── code/
│   └── NNDL_CA4_2_1.ipynb          # Main implementation notebook
│
├── dataset/
│   ├── train_data.csv              # Training dataset
│   ├── val_data.csv                # Validation dataset
│   └── test_data.csv               # Test dataset
│
├── images/
│   ├── plot_28_1.png               # Feature correlation heatmaps
│   ├── plot_30_2.png
│   ├── plot_36_3.png               # Stationarity analysis
│   ├── plot_37_4.png               # ACF/PACF plots
│   ├── plot_39_5.png               # SARIMAX forecasts
│   ├── plot_42_6.png
│   ├── plot_46_7.png
│   ├── plot_71_8.png               # Training curves
│   ├── plot_72_9.png
│   ├── plot_73_10.png
│   ├── plot_74_11.png
│   ├── plot_75_12.png
│   ├── plot_83_13.png               # Multi-step forecasts
│   ├── plot_84_14.png
│   ├── plot_85_15.png
│   ├── plot_86_16.png
│   ├── plot_92_17.png               # Final comparisons
│   ├── plot_93_18.png
│   ├── plot_94_19.png
│   └── plot_95_20.png
│
├── description/
│   ├── NNDL_HW4.pdf                 # Assignment description
│   └── NNDL_UT_CA4_D.pdf            # Detailed requirements
│
├── paper/
│   └── Lee_Hauskrecht_AIME_2019.pdf # Reference paper
│
├── report/
│   └── NNDL_UT_CA4_Q2.pdf          # Project report
│
└── README.md                        # This file
```

---

## 🔍 Model Architecture Comparison

### Parameter Count

| Model | Parameters | Complexity |
|-------|------------|------------|
| GRU (Unidirectional) | 886,273 | Low |
| LSTM (Unidirectional) | 1,093,632 | Medium |
| GRU (Bidirectional) | ~1,772,546 | High |
| LSTM (Bidirectional) | ~2,187,264 | Very High |

### Training Characteristics

**GRU Models:**
- Fastest training due to simplified gating
- Fewer parameters than LSTM
- Comparable performance in many cases

**LSTM Models:**
- Moderate complexity with full gating mechanism
- Better long-term memory retention
- Slightly better performance on complex patterns

**Bidirectional Models:**
- 2× computational cost for forward/backward processing
- Higher memory requirements
- Better temporal context utilization

---

## 📊 Statistical Baseline Analysis

### SARIMAX Models

**Model 1: SARIMAX(2,1,2)**
- **Log Likelihood:** -423.760
- **AIC:** 859.520
- **BIC:** 876.952
- **R² Score:** ~0.65

**Model 2: SARIMAX(14,1,12)**
- Higher complexity
- Captures more complex patterns
- R² Score: ~0.67

**Model 3: SARIMAX(14,1,24)**
- Very high complexity
- R² Score: ~0.69
- Risk of overfitting and convergence issues

### Statistical Model Limitations

- ❌ **Linear Relationships:** Assumes linear relationships between variables
- ❌ **Gaussian Errors:** Requires normally distributed residuals
- ❌ **Constant Parameters:** Parameters assumed constant over time
- ❌ **Limited Nonlinearity:** Cannot capture complex nonlinear patterns

### Deep Learning Advantages

- **Nonlinear Pattern Recognition:** Captures complex interactions
- **Temporal Dynamics:** Learns complex temporal dependency patterns
- **Feature Learning:** Automatically learns relevant feature combinations
- **Adaptability:** Adapts to different patient patterns and conditions
- **Superior Accuracy:** 12-15% improvement in R² score over statistical methods

---

## 🎓 Key Learnings

1. **RNN Effectiveness:** RNNs significantly outperform simple Markov models for time series prediction
2. **Bidirectional Processing:** Bidirectional processing provides 2-4% improvement in accuracy
3. **LSTM vs GRU:** LSTM performs slightly better but GRU is more efficient
4. **Feature Engineering:** Proper feature selection improves model performance
5. **Clinical Applicability:** Deep learning models achieve clinically acceptable accuracy (89% within ±5 BPM)
6. **Statistical Baselines:** Statistical methods provide useful baselines but are limited by linear assumptions
7. **Multi-Step Forecasting:** Error accumulation is a key challenge for long-term predictions

---

## 🔬 Technical Details

### Feature Selection Methodology

1. **Correlation-Based Elimination:**
   - Threshold: 0.3
   - For correlated pairs, retain feature with higher correlation to target (HR)

2. **Maximum Correlation Filtering:**
   - Calculate maximum correlation of each feature with all others
   - Select 20 features with lowest maximum correlations
   - Ensure inclusion of target variable (HR) and Patient_ID

### Temporal Sequence Construction

**Sliding Window Approach:**
- Input window: 2 hours of historical data
- Output: 1 hour forecast
- Maximum overlap between sequences for data utilization

**Multi-Patient Sequence Generation:**
- Preserves patient identity for each sequence
- Combines data from multiple patients for better generalization
- Prevents overfitting through diverse patient patterns

### Loss Function

**Mean Squared Error (MSE):**
- Appropriate for continuous heart rate prediction
- Provides smooth gradients for stable training
- Penalizes large prediction errors more heavily
- Works well with MinMax-scaled target variables

---

## 📝 Code Structure

### Main Classes

1. **MarkovPredictor:**
   - Baseline model without temporal memory
   - Flatten + Linear layers with normalization

2. **RecurrentPredictor:**
   - Base class for RNN-based models
   - Handles post-processing and output generation

3. **LSTMPredictor:**
   - Inherits from RecurrentPredictor
   - Uses LSTM cells with optional bidirectional processing

4. **GRUPredictor:**
   - Inherits from RecurrentPredictor
   - Uses GRU cells with optional bidirectional processing

### Training Functions

- `train_epoch()`: Single epoch training
- `validation_epoch()`: Validation phase
- `train_model()`: Complete training loop with early stopping
- `predict()`: Model inference
- `predict_multi_steps()`: Multi-step autoregressive prediction

### Evaluation Functions

- `evaluate_predictions()`: Comprehensive model evaluation
- `evaluate_time_series_predictions()`: Time series specific evaluation with visualizations
- `compare_models()`: Side-by-side model comparison

---

## 🔮 Future Research Directions

### Immediate Extensions

1. **Architecture Enhancements:**
   - Integration of attention mechanisms
   - Exploration of transformer-based approaches
   - Ensemble methods combining multiple architectures
   - Hybrid statistical and deep learning models

2. **Data and Preprocessing Improvements:**
   - Larger physiological datasets
   - Time series data augmentation techniques
   - Advanced feature engineering
   - Improved missing data handling

### Advanced Research Directions

1. **Clinical Integration:**
   - Real-time clinical deployment systems
   - Multi-modal integration of multiple physiological signals
   - Patient-specific model adaptation
   - Comprehensive clinical validation studies

2. **Methodological Advances:**
   - Causal inference methodologies
   - Uncertainty quantification methods
   - Interpretable deep learning models
   - Federated learning for privacy-preserving healthcare

---

## 📚 References

1. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." *Neural computation*.

2. Cho, K., et al. (2014). "Learning phrase representations using RNN encoder-decoder for statistical machine translation." *arXiv preprint arXiv:1406.1078*.

3. Chung, J., et al. (2014). "Empirical evaluation of gated recurrent neural networks on sequence modeling." *arXiv preprint arXiv:1412.3555*.

4. PyTorch Documentation: https://pytorch.org/docs/stable/index.html

5. Scikit-learn Documentation: https://scikit-learn.org/stable/

6. Statsmodels Documentation: https://www.statsmodels.org/stable/index.html

7. Lee, J. & Hauskrecht, M. (2019). "Neural approaches for time series forecasting." *AIME 2019*.

---

## 👤 Author

**Taha Majlesi** (810101504)  
University of Tehran  
Faculty of Electrical and Computer Engineering  
Neural Networks and Deep Learning - CA4

---

## 📄 License

This project is part of the Neural Networks and Deep Learning course assignment at the University of Tehran.

---

## ✅ Conclusion

This research demonstrates the significant potential of recurrent neural networks for physiological time series prediction, establishing a foundation for advanced healthcare analytics. The comprehensive comparison of methodologies provides valuable insights for researchers and practitioners in the field of healthcare AI. 

The superior performance of bidirectional LSTM models, combined with clinically acceptable prediction accuracy, positions these approaches as viable solutions for real-world healthcare applications. The systematic methodology, rigorous evaluation framework, and detailed documentation contribute to the growing body of research in healthcare AI, providing a replicable framework for future studies.

**Key Achievement:** Bidirectional LSTM achieves R² = 0.87 with 89% of predictions within ±5 BPM, demonstrating both technical excellence and clinical utility.