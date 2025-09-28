import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
from scipy.stats import pearsonr


def evaluate_time_series_predictions(y_true, y_pred, model_name="Model"):
    """Comprehensive time series prediction evaluation."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_var = explained_variance_score(y_true, y_pred)

    corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())

    print(f"{model_name} Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Explained Variance: {explained_var:.4f}")
    print(f"Pearson Correlation: {corr:.4f}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(y_true[:100], label="True", alpha=0.7)
    plt.plot(y_pred[:100], label="Predicted", alpha=0.7)
    plt.title(f"{model_name}: True vs Predicted (First 100 samples)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    residuals = y_true - y_pred
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred[:100], residuals[:100], alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title(f"{model_name}: Residual Plot")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=50, alpha=0.7, edgecolor="black")
    plt.title(f"{model_name}: Residual Distribution")
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "explained_variance": explained_var,
        "correlation": corr,
    }


def compare_models(results_dict):
    """Compare multiple models side by side."""
    models = list(results_dict.keys())
    metrics = ["rmse", "mae", "r2", "explained_variance", "correlation"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))

    for i, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        bars = axes[i].bar(models, values)
        axes[i].set_title(f"{metric.upper()}")
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis="x", rotation=45)

        for bar, value in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import cosine
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torchinfo import summary
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import defaultdict
from statsmodels.tsa.stattools import adfuller
import torch
import cv2
import os
import math
import re


from google.colab import drive
drive.mount('/content/drive/')


class MarkovPredictor(nn.Module):
  def __init__(self, num_features, window_size, forecast_horizon, transition_size):
    super().__init__()
    self.flatten = nn.Flatten(1)
    self.linear_in = nn.Linear(window_size*num_features,window_size*transition_size)
    self.norm = nn.BatchNorm1d(window_size*transition_size)
    self.dropout = nn.Dropout(0.5)
    self.linear_out = nn.Linear(window_size*transition_size, forecast_horizon)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.flatten(x)
    x = self.linear_in(x)
    x = self.norm(x)
    x = self.dropout(x)
    x = self.linear_out(x)
    return self.sigmoid(x)


class RecurrentPredictor(nn.Module):
  def __init__(self, forecast_horizon, hidden_state_size, fully_connected_size, rnn):
    super(RecurrentPredictor, self).__init__()
    self.rnn = rnn
    self.linear = nn.Linear(hidden_state_size, fully_connected_size)
    self.norm = nn.BatchNorm1d(fully_connected_size)
    self.dropout = nn.Dropout(0.5)
    self.linear_out = nn.Linear(fully_connected_size, forecast_horizon)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x,_ = self.rnn(x)
    x = x[:,-1,:]
    x = self.linear(x)
    x = self.norm(x)
    x = self.dropout(x)
    x = self.linear_out(x)
    return self.sigmoid(x)


class GRUPredictor(RecurrentPredictor):
  def __init__(self, num_features, bidirectional, forecast_horizon, hidden_state_size, fully_connected_size):
    rnn = nn.GRU(
        input_size = num_features,
        hidden_size  = hidden_state_size,
        batch_first = True,
        bidirectional = bidirectional,
    )
    if bidirectional:
      hidden_state_size *= 2
    super().__init__(forecast_horizon, hidden_state_size, fully_connected_size, rnn)


class LSTMPredictor(RecurrentPredictor):
  def __init__(self, num_features, bidirectional, forecast_horizon, hidden_state_size, fully_connected_size):
    rnn = nn.LSTM(
        input_size = num_features,
        hidden_size  = hidden_state_size,
        batch_first = True,
        bidirectional = bidirectional,
    )
    if bidirectional:
      hidden_state_size *= 2
    super().__init__(forecast_horizon, hidden_state_size, fully_connected_size, rnn)


path = '/content/drive/MyDrive/Colab/NNDL/CA4/Part1/Dataset/'
df_train = pd.read_csv(os.path.join(path,"train_data.csv"))
df_val = pd.read_csv(os.path.join(path,"val_data.csv"))
df_test = pd.read_csv(os.path.join(path,"test_data.csv"))


df_train.head(5).round(2)


len(df_train),len(df_val),len(df_test)


df_train.nunique()


df_train.info()


df_train.describe()


main_columns_idx = [2,43]
target_idx = 2
main_columns_name = ['HR','Patient_ID']
columns = [True for col in df_val.columns]
columns[0] = False
columns[1] = False
columns[41] = False
threshold = 0.3
corr = df_val.corr().abs()

for i in range(corr.shape[0]):
  for j in range(i+1, corr.shape[0]):
    if i in main_columns_idx or j in main_columns_idx:
      continue
    if columns[j] == False or columns[i] == False:
      continue
    if corr.iloc[i,j] >= threshold:
      if corr.iloc[i,target_idx]>corr.iloc[j,target_idx]:
        columns[j] = False
      else:
        columns[i] = False

len(df_val.columns[columns])


(df_train['Unnamed: 0']==df_train['Hour']).sum()


columns[0] = False
for col in main_columns_idx:
  columns[col] = True
chosen_cols = df_val.columns[columns]
chosen_cols


corr = df_val[chosen_cols].corr()


plt.figure(figsize=(18,18))
sns.heatmap(corr.round(2), annot=True, vmin=-1, vmax=1, cmap='Blues')


max_corrs = []
n_features = 20

maxs = corr.drop(main_columns_name).abs()
maxs = maxs[maxs<1].max()
for i,name in enumerate(corr):
  if name in main_columns_name:
    continue
  max_corrs.append((maxs.iloc[i],i))
sorted_corrs = sorted(max_corrs, key=lambda x: -x[0])

columns  = [True for col in corr.columns]
for i in range(len(sorted_corrs)-n_features+len(main_columns_name)-1):
  columns[sorted_corrs[i][1]] = False

chosen_cols = corr.columns[columns]
len(chosen_cols)


corr = df_val[chosen_cols].corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr.round(2), annot=True, vmin=-1, vmax=1, cmap='Blues')


def get_grouped_df(df,chosen_cols,group_col):
  clean_cols = [col for col in chosen_cols if col != group_col]
  default = {col:[] for col in clean_cols}
  def df_factory():
      return pd.DataFrame(default)

  new_df = defaultdict(df_factory)
  for x in list(df[chosen_cols].groupby(group_col)):
    new_df[x[0]] = x[1][clean_cols]

  return new_df


group_col = 'Patient_ID'
train = get_grouped_df(df_train,chosen_cols,group_col)
val = get_grouped_df(df_val,chosen_cols,group_col)
test = get_grouped_df(df_test,chosen_cols,group_col)


id = list(train.keys())[0]
df = pd.concat((train[id],val[id],test[id])).reset_index(drop=True)
df


hr_series = df['HR']
d = 0
adf_result = adfuller(hr_series)
p_value = adf_result[1]
print(f'd = {d}, test result = {p_value}')

while p_value >= 0.05:
    d += 1
    hr_series = hr_series.diff().dropna()
    adf_result = adfuller(hr_series)
    p_value = adf_result[1]
    print(f'd = {d}, test result = {p_value}')


plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(df['HR'])
plt.title('Heart rate of patient 9')

plt.subplot(122)
plt.plot(hr_series)
plt.title('Differentiated heart rate of patient 9')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))
plt.subplot(121)
plot_acf(hr_series, ax=plt.gca(), lags=50)
plt.title('ACF - Determine q (MA order)')

plt.subplot(122)
plot_pacf(hr_series, ax=plt.gca(), lags=50, method='ywm')
plt.title('PACF - Determine p (AR order)')
plt.tight_layout()
plt.show()


traindf = train[id].reset_index(drop=True)
testdf = val[id].reset_index(drop=True)

p, q = 2, 2
model1 = SARIMAX(traindf['HR'],
                order=(p, d, q),
                exog=traindf['O2Sat'],
)
model_fit1 = model1.fit(disp=False)

print(model_fit1.summary())


forecast = model_fit1.forecast(steps=len(testdf),exog=testdf['O2Sat'])
plt.figure(figsize=(10, 5))
plt.plot(traindf['HR'], label='Train HR')
plt.plot(range(len(traindf), len(traindf) + len(testdf)), testdf['HR'], label='True Val HR')
plt.plot(range(len(traindf), len(traindf) + len(testdf)), forecast, label='Predicted HR')
plt.xlabel('Time Index')
plt.ylabel('Heart Rate')
plt.title('HR Forecast vs True HR')
plt.legend()
plt.show()


r2_score(testdf['HR'],forecast)


traindf = train[id].reset_index(drop=True)
testdf = val[id].reset_index(drop=True)

p, q = 14,12
model2 = SARIMAX(traindf['HR'],
                order=(p, d, q),
                exog=traindf['O2Sat'],
)
model_fit2 = model2.fit(disp=False)

print(model_fit2.summary())


forecast = model_fit2.forecast(steps=len(testdf),exog=testdf['O2Sat'])
plt.figure(figsize=(10, 5))
plt.plot(traindf['HR'], label='Train HR')
plt.plot(range(len(traindf), len(traindf) + len(testdf)), testdf['HR'], label='True Val HR')
plt.plot(range(len(traindf), len(traindf) + len(testdf)), forecast, label='Predicted HR')
plt.ylim(20,150)
plt.xlabel('Time Index')
plt.ylabel('Heart Rate')
plt.title('HR Forecast vs True HR')
plt.legend()
plt.show()


r2_score(testdf['HR'],forecast.reset_index(drop=True))


print(testdf['HR']-forecast.reset_index(drop=True))


traindf = train[id].reset_index(drop=True)
testdf = val[id].reset_index(drop=True)

p, q = 14,24
model3 = SARIMAX(traindf['HR'],
                order=(p, d, q),
                exog=traindf['O2Sat'],
)
model_fit3 = model3.fit(disp=False)

print(model_fit3.summary())


forecast = model_fit3.forecast(steps=len(testdf),exog=testdf['O2Sat'])
plt.figure(figsize=(10, 5))
plt.plot(traindf['HR'], label='Train HR')
plt.plot(range(len(traindf), len(traindf) + len(testdf)), testdf['HR'], label='True Val HR')
plt.plot(range(len(traindf), len(traindf) + len(testdf)), forecast, label='Predicted HR')
plt.ylim(20,150)
plt.xlabel('Time Index')
plt.ylabel('Heart Rate')
plt.title('HR Forecast vs True HR')
plt.legend()
plt.show()


r2_score(testdf['HR'],forecast.reset_index(drop=True))


print(testdf['HR']-forecast.reset_index(drop=True))


scaler = MinMaxScaler()
cols = [col for col in chosen_cols if col not in ['HR','Patient_ID']]

scaled_train = df_train.copy()
scaled_train[cols] = scaler.fit_transform(df_train[cols])
scaled_train[cols] = pd.DataFrame(scaled_train, columns=cols)

scaled_val = df_val.copy()
scaled_val[cols] = scaler.transform(df_val[cols])
scaled_val[cols] = pd.DataFrame(scaled_val, columns=cols)

scaled_test = df_test.copy()
scaled_test[cols] = scaler.transform(df_test[cols])
scaled_test[cols] = pd.DataFrame(scaled_test, columns=cols)


HRScaler = MinMaxScaler()
targets = ['HR']
scaled_train[targets] = HRScaler.fit_transform(scaled_train[targets])
scaled_val[targets] = HRScaler.transform(scaled_val[targets])
scaled_test[targets] = HRScaler.transform(scaled_test[targets])


group_col = 'Patient_ID'
train = get_grouped_df(scaled_train,chosen_cols,group_col)
val = get_grouped_df(scaled_val,chosen_cols,group_col)
test = get_grouped_df(scaled_test,chosen_cols,group_col)


def create_window(df, input_size, output_size,target):
  xs = []
  ys = []

  for i in range(len(df) - input_size - output_size + 1):
    x = df[i:(i + input_size)].to_numpy()
    y = df[i + input_size:i + input_size + output_size][target].to_numpy()
    xs.append(x)
    ys.append(y)

  return xs, ys

def create_data_for_each_person(data, input_size, output_size,target):
  xs = []
  ys = []

  for id in list(data.keys()):
    x,y = create_window(data[id],input_size,output_size,target)
    xs.extend(x)
    ys.extend(y)

  return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


input_size = 2
output_size = 1
target = 'HR'

X_train,y_train = create_data_for_each_person(train, input_size, output_size,target)
X_val,y_val = create_data_for_each_person(val, input_size, output_size,target)
X_test,y_test = create_data_for_each_person(test, input_size, output_size,target)


X_train.shape, y_train.shape, X_val.shape, X_test.shape


num_features = 20
window_size = 2
forecast_horizon = 1
transition_size = 128

markov = MarkovPredictor(num_features, window_size, forecast_horizon, transition_size)


summary(markov, (1,2,20))


hidden_state_size = 512
fully_connected_size = 128

gru = GRUPredictor(num_features, False, forecast_horizon, hidden_state_size, fully_connected_size)
lstm = LSTMPredictor(num_features, False, forecast_horizon, hidden_state_size, fully_connected_size)
bidir_gru = GRUPredictor(num_features, True, forecast_horizon, hidden_state_size, fully_connected_size)
bidir_lstm = LSTMPredictor(num_features, True, forecast_horizon, hidden_state_size, fully_connected_size)


print(summary(gru, (1,2,20)))
print(summary(lstm, (1,2,20)))
print(summary(bidir_gru, (1,2,20)))
print(summary(bidir_lstm, (1,2,20)))


def train_epoch(model, data_loader, criterion, optimizer, device):
  model.train()
  total_loss = 0
  num_batches = len(data_loader)

  for x,y in data_loader:
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()
    y_preds = model(x)
    loss = criterion(y_preds, y)

    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  return total_loss / num_batches

def validation_epoch(model, data_loader, criterion, device):
  model.eval()
  total_loss = 0
  num_batches = len(data_loader)

  with torch.no_grad():
    for x,y in data_loader:
      x = x.to(device)
      y = y.to(device)

      y_preds = model(x)
      loss = criterion(y_preds, y)
      total_loss += loss.item()

  return total_loss / num_batches


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, do_early_stopping=False, patience=0):
  hist = {
      "train_loss": [],
      "val_loss": [],
  }

  best_val_loss = float('inf')
  model = model.to(device)
  epochs_no_improve = 0

  for epoch in range(1, epochs + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    hist['train_loss'].append(train_loss)
    print(f"Epoch [{epoch}] Average Training Loss: {train_loss:.4f}")

    val_loss = validation_epoch(model, val_loader, criterion, device)
    hist['val_loss'].append(val_loss)
    print(f"Epoch [{epoch}] Average Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience and do_early_stopping:
            print(f"Early stopping at epoch: {epoch}")
            model.load_state_dict(torch.load('best_model.pth'))
            return hist

  model.load_state_dict(torch.load('best_model.pth'))
  return hist


num_features = 20
window_size = 2
forecast_horizon = 1
transition_size = 128


X_train,y_train = create_data_for_each_person(train, window_size, forecast_horizon,target)
X_val,y_val = create_data_for_each_person(val, window_size, forecast_horizon,target)
X_test,y_test = create_data_for_each_person(test, window_size, forecast_horizon,target)


X_train.shape, y_train.shape, X_val.shape, X_test.shape


dataset_train = TensorDataset(X_train, y_train)
dataset_val = TensorDataset(X_val, y_val)
dataset_test = TensorDataset(X_test, y_test)


epochs = 50
batch_size = 32
learning_rate = 0.005
patience = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


loader_train = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True
)
loader_val = DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False
)
loader_test = DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False
)


markov = MarkovPredictor(num_features, window_size, forecast_horizon, transition_size)
optimizer = torch.optim.SGD(markov.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
hist_markov = train_model(markov,loader_train, loader_val, criterion, optimizer, epochs, device, do_early_stopping=True,patience = patience)


def plot_history(history):
    epochs = len(history['train_loss'])
    plt.plot(range(1,epochs+1), history['train_loss'])
    plt.plot(range(1,epochs+1), history['val_loss'])
    plt.title('Loss of model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation/Test'])
    plt.ylim(0, max(max(history['train_loss']),max(history['val_loss']))*1.05)
    plt.show()


plot_history(hist_markov)


gru = GRUPredictor(num_features, False, forecast_horizon, hidden_state_size, fully_connected_size)
optimizer = torch.optim.SGD(gru.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
hist_gru = train_model(gru,loader_train, loader_val, criterion, optimizer, epochs, device, do_early_stopping=True,patience = patience)
plot_history(hist_gru)


lstm = LSTMPredictor(num_features, False, forecast_horizon, hidden_state_size, fully_connected_size)
optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
hist_lstm = train_model(lstm,loader_train, loader_val, criterion, optimizer, epochs, device, do_early_stopping=True,patience = patience)
plot_history(hist_lstm)


bidir_gru = GRUPredictor(num_features, True, forecast_horizon, hidden_state_size, fully_connected_size)
optimizer = torch.optim.SGD(bidir_gru.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
hist_bidir_gru = train_model(bidir_gru,loader_train, loader_val, criterion, optimizer, epochs, device, do_early_stopping=True,patience = patience)
plot_history(hist_bidir_gru)


bidir_lstm = LSTMPredictor(num_features, True, forecast_horizon, hidden_state_size, fully_connected_size)
optimizer = torch.optim.SGD(bidir_lstm.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
hist_bidir_lstm = train_model(bidir_lstm,loader_train, loader_val, criterion, optimizer, epochs, device, do_early_stopping=True,patience = patience)
plot_history(hist_bidir_lstm)


def evaluate_predictions(y_true, y_pred, res_dict, label):
    evs = explained_variance_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    cosine_dist = cosine(y_true, y_pred)

    res_dict['Explained Variance Score'][label] = evs
    res_dict['Mean Squared Error'][label] = mse
    res_dict['Mean Absolute Error'][label] = mae
    res_dict['R2 Score'][label] = r2
    res_dict['Cosine Distance'][label] = cosine_dist

    return res_dict


def predict(model, data_loader, device):
  model.eval()
  total_loss = 0
  num_batches = len(data_loader)
  y_true = []
  preds = []

  with torch.no_grad():
    for x,y in data_loader:
      x = x.to(device)
      y_true.append(y.cpu().numpy())
      y = y.to(device)

      y_preds = model(x)
      preds.append(y_preds.cpu().numpy())

  y_true = np.concatenate(y_true, axis=0).flatten()
  preds = np.concatenate(preds, axis=0).flatten()

  return y_true, preds


res_dict = defaultdict(lambda: defaultdict(float))

y_true, preds = predict(markov, loader_val, device)
res_dict = evaluate_predictions(y_true, preds, res_dict, 'Dense')

y_true, preds = predict(gru, loader_val, device)
res_dict = evaluate_predictions(y_true, preds, res_dict, 'GRU')

y_true, preds = predict(lstm, loader_val, device)
res_dict = evaluate_predictions(y_true, preds, res_dict, 'LSTM')

y_true, preds = predict(bidir_gru, loader_val, device)
res_dict = evaluate_predictions(y_true, preds, res_dict, 'Bidirectional GRU')

y_true, preds = predict(bidir_lstm, loader_val, device)
res_dict = evaluate_predictions(y_true, preds, res_dict, 'Bidirectional LSTM')

res_dict = pd.DataFrame(res_dict).T
res_dict


def predict_multi_steps(model,inputs,steps,window_size,horizon,device):
  start = 0
  sequence = []
  inputs = inputs.values.tolist()
  len_inputs = len(inputs)
  model = model.to(device)
  model.eval()
  idx = 0
  done = False

  with torch.no_grad():
    while(len(sequence)<steps and done == False):
      if idx + window_size > len(inputs):
        window = inputs[-window_size:]
        done = True
      else:
        window = inputs[idx:idx+window_size]
      idx += horizon
      x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
      preds = model(x)
      preds = preds.cpu().tolist()[0]
      sequence.extend(preds)

  out = np.array(sequence[:steps]).reshape(1, -1)
  return HRScaler.inverse_transform(out)[0]


df = pd.concat((train[id],val[id],test[id])).reset_index(drop=True)
train_hr = df[len(train[id])-window_size:]


forecast1 = model_fit1.forecast(steps=len(testdf),exog=testdf['O2Sat'])
forecast2 = model_fit3.forecast(steps=len(testdf),exog=testdf['O2Sat'])
markov_preds = predict_multi_steps(markov,train_hr,len(testdf),window_size,forecast_horizon,device)
gru_preds = predict_multi_steps(gru,train_hr,len(testdf),window_size,forecast_horizon,device)
lstm_preds = predict_multi_steps(lstm,train_hr,len(testdf),window_size,forecast_horizon,device)
bidir_gru_preds = predict_multi_steps(bidir_gru,train_hr,len(testdf),window_size,forecast_horizon,device)
bidir_lstm_preds = predict_multi_steps(bidir_lstm,train_hr,len(testdf),window_size,forecast_horizon,device)

plt.figure(figsize=(14, 7), dpi=100)
plt.plot(testdf['HR'].values, label='True HR', color='black', linewidth=2)
plt.scatter(range(len(testdf)), forecast1, label='SARIMAX p=2 q=2', s=25, marker='o')
plt.scatter(range(len(testdf)), forecast2, label='SARIMAX p=14 q=24', s=25, marker='o')
plt.scatter(range(len(testdf)), markov_preds, label='Dense (Markov)', s=25, marker='x')
plt.scatter(range(len(testdf)), gru_preds, label='GRU', s=25, marker='^')
plt.scatter(range(len(testdf)), lstm_preds, label='LSTM', s=25, marker='s')
plt.scatter(range(len(testdf)), bidir_gru_preds, label='Bidirectional GRU', s=25, marker='P')
plt.scatter(range(len(testdf)), bidir_lstm_preds, label='Bidirectional LSTM', s=25, marker='D')
plt.ylim(20,150)
plt.xlabel('Time Index')
plt.ylabel('Heart Rate')
plt.title('HR Forecast of Models')
plt.legend()
plt.show()


plt.figure(figsize=(14, 7), dpi=100)
plt.plot(testdf['HR'].values, label='True HR', color='black', linewidth=2)
plt.scatter(range(len(testdf)), forecast1, label='SARIMAX p=2 q=2', s=25, marker='o')
plt.scatter(range(len(testdf)), forecast2, label='SARIMAX p=14 q=24', s=25, marker='o')
plt.scatter(range(len(testdf)), markov_preds, label='Dense (Markov)', s=25, marker='x')
plt.ylim(20,150)
plt.xlabel('Time Index')
plt.ylabel('Heart Rate')
plt.title('HR Forecast of Models')
plt.legend()
plt.show()


plt.figure(figsize=(14, 7), dpi=100)
plt.plot(testdf['HR'].values, label='True HR', color='black', linewidth=2)
plt.scatter(range(len(testdf)), forecast1, label='SARIMAX p=2 q=2', s=25, marker='o')
plt.scatter(range(len(testdf)), gru_preds, label='GRU', s=25, marker='^')
plt.scatter(range(len(testdf)), lstm_preds, label='LSTM', s=25, marker='s')
plt.ylim(20,150)
plt.xlabel('Time Index')
plt.ylabel('Heart Rate')
plt.title('HR Forecast of Models')
plt.legend()
plt.show()


plt.figure(figsize=(14, 7), dpi=100)
plt.plot(testdf['HR'].values, label='True HR', color='black', linewidth=2)
plt.scatter(range(len(testdf)), forecast1, label='SARIMAX p=2 q=2', s=25, marker='o')
plt.scatter(range(len(testdf)), bidir_gru_preds, label='Bidirectional GRU', s=25, marker='P')
plt.scatter(range(len(testdf)), bidir_lstm_preds, label='Bidirectional LSTM', s=25, marker='D')
plt.ylim(20,150)
plt.xlabel('Time Index')
plt.ylabel('Heart Rate')
plt.title('HR Forecast of Models')
plt.legend()
plt.show()


res_dict = defaultdict(lambda: defaultdict(float))
y_true = testdf['HR'].values

res_dict = evaluate_predictions(y_true, forecast1, res_dict, 'SARIMAX p=2 q=2')
res_dict = evaluate_predictions(y_true, forecast2, res_dict, 'SARIMAX p=14 q=25')
res_dict = evaluate_predictions(y_true, markov_preds, res_dict, 'Dense')
res_dict = evaluate_predictions(y_true, gru_preds, res_dict, 'GRU')
res_dict = evaluate_predictions(y_true, lstm_preds, res_dict, 'LSTM')
res_dict = evaluate_predictions(y_true, bidir_gru_preds, res_dict, 'Bidirectional GRU')
res_dict = evaluate_predictions(y_true, bidir_lstm_preds, res_dict, 'Bidirectional LSTM')

res_dict = pd.DataFrame(res_dict).T
res_dict


train_val_dataset = ConcatDataset([dataset_train, dataset_val])
loader_train_val = DataLoader(
    train_val_dataset,
    batch_size=batch_size,
    shuffle=True)


epochs = 10


markov = MarkovPredictor(num_features, window_size, forecast_horizon, transition_size)
optimizer = torch.optim.SGD(markov.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
hist_markov = train_model(markov,loader_train_val, loader_test, criterion, optimizer, epochs, device, do_early_stopping=False)


gru = GRUPredictor(num_features, False, forecast_horizon, hidden_state_size, fully_connected_size)
optimizer = torch.optim.SGD(gru.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
hist_gru = train_model(gru,loader_train_val, loader_test, criterion, optimizer, epochs, device, do_early_stopping=False)
plot_history(hist_gru)


lstm = LSTMPredictor(num_features, False, forecast_horizon, hidden_state_size, fully_connected_size)
optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
hist_lstm = train_model(lstm,loader_train_val, loader_test, criterion, optimizer, epochs, device, do_early_stopping=False)
plot_history(hist_lstm)


bidir_gru = GRUPredictor(num_features, True, forecast_horizon, hidden_state_size, fully_connected_size)
optimizer = torch.optim.SGD(bidir_gru.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
hist_bidir_gru = train_model(bidir_gru,loader_train_val, loader_test, criterion, optimizer, epochs, device, do_early_stopping=False)
plot_history(hist_bidir_gru)


bidir_lstm = LSTMPredictor(num_features, True, forecast_horizon, hidden_state_size, fully_connected_size)
optimizer = torch.optim.SGD(bidir_lstm.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
hist_bidir_lstm = train_model(bidir_lstm,loader_train_val, loader_test, criterion, optimizer, epochs, device, do_early_stopping=False)
plot_history(hist_bidir_lstm)


res_dict = defaultdict(lambda: defaultdict(float))

y_true, preds = predict(markov, loader_test, device)
res_dict = evaluate_predictions(y_true, preds, res_dict, 'Dense')

y_true, preds = predict(gru, loader_test, device)
res_dict = evaluate_predictions(y_true, preds, res_dict, 'GRU')

y_true, preds = predict(lstm, loader_test, device)
res_dict = evaluate_predictions(y_true, preds, res_dict, 'LSTM')

y_true, preds = predict(bidir_gru, loader_test, device)
res_dict = evaluate_predictions(y_true, preds, res_dict, 'Bidirectional GRU')

y_true, preds = predict(bidir_lstm, loader_test, device)
res_dict = evaluate_predictions(y_true, preds, res_dict, 'Bidirectional LSTM')

res_dict = pd.DataFrame(res_dict).T
res_dict

