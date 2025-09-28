from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def evaluate_fraud_detection(y_true, y_pred_proba, y_pred=None, threshold=0.5):
    """Comprehensive evaluation for fraud detection."""
    if y_pred is None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-Fraud', 'Fraud'], digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Fraud', 'Fraud'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    disp.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix')
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ax2.plot(fpr, tpr, label=f'AUC = {auc_roc:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'auc_roc': auc_roc,
        'precision': cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
        'recall': cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0,
        'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    }



from google.colab import files
files.upload()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential






df = pd.read_csv('creditcard.csv')

print("Data Head:")
print(df.head())
print("\nData Info:")
df.info()
print("\nData Description:")
print(df.describe())




plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
plt.ylabel('Count')
plt.show()


class_counts = df['Class'].value_counts()
print(f"Non-Fraudulent transactions: {class_counts[0]}")
print(f"Fraudulent transactions: {class_counts[1]}")
print(f"Percentage of fraudulent transactions: {class_counts[1] / len(df) * 100:.4f}%")



scaler = StandardScaler()
df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1)





X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)



print("\nShape of training data (X_train):", X_train.shape)
print("Shape of testing data (X_test):", X_test.shape)
print("Distribution of classes in training set:\n", y_train.value_counts(normalize=True))
print("Distribution of classes in testing set:\n", y_test.value_counts(normalize=True))



def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def evaluate_model(model, X_test, y_test, model_name):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(f"\n--- Evaluation for {model_name} ---")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

def create_mlp_model(neurons=64, dropout_rate=0.2, reg_lambda=0.001):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=INPUT_SHAPE, kernel_regularizer=l2(reg_lambda)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


EPOCHS = 40
BATCH_SIZE = 32
INPUT_SHAPE = (X_train.shape[1],)


simple_mlp = Sequential([
    Dense(64, activation='relu', input_shape=INPUT_SHAPE),
    Dense(1, activation='sigmoid')
])
simple_mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_simple = simple_mlp.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=0)
plot_history(history_simple, "Simple MLP")
evaluate_model(simple_mlp, X_test, y_test, "Simple MLP")


model_for_grid = KerasClassifier(build_fn=create_mlp_model, epochs=10, verbose=0)

param_grid = {
    'batch_size': [16, 32, 64],
    'model__neurons': [64, 128, 256],
    'model__dropout_rate': [0.2, 0.3, 0.4],
    'model__reg_lambda': [0.001, 0.0001]
}


grid = GridSearchCV(estimator=model_for_grid, param_grid=param_grid, cv=3, scoring='f1', n_jobs=1)
grid_result = grid.fit(X_train, y_train)

print(f"Best F1-Score: {grid_result.best_score_:.4f} using {grid_result.best_params_}")


best_params = grid_result.best_params_
best_model = create_mlp_model(neurons=best_params['neurons'],
                              dropout_rate=best_params['dropout_rate'],
                              reg_lambda=best_params['reg_lambda'])

history_best = best_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=best_params['batch_size'], validation_split=0.2, verbose=0)
plot_history(history_best, "Best MLP from Grid Search")
evaluate_model(best_model, X_test, y_test, "Best MLP from Grid Search")


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)


y_pred_lr = lr_model.predict(X_test)
y_pred_prob_lr = lr_model.predict_proba(X_test)[:, 1]


cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")


fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_lr:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()



url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
df = pd.read_excel(url)

df.columns = ['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 'Superplasticizer',
              'CoarseAggregate', 'FineAggregate', 'Age', 'ConcreteCompressiveStrength']



corr_with_target = correlation_matrix['ConcreteCompressiveStrength'].sort_values(ascending=False)
highest_corr_feature = corr_with_target.index[1]
print(f"\n4. The feature with the highest correlation with concrete strength is: '{highest_corr_feature}' with a value of {corr_with_target[1]:.2f}.")



strong_inter_corr = correlation_matrix[(abs(correlation_matrix) > 0.6) & (correlation_matrix < 1.0)]
if df[['Superplasticizer', 'Water']].corr().iloc[0,1] < -0.6:
    print("- 'Superplasticizer' and 'Water' have a strong negative correlation, which is logical as superplasticizers are used to reduce water content.")
else:
    print("- No very strong correlations between predictor features were identified based on a 0.6 threshold.")




X = df.drop('ConcreteCompressiveStrength', axis=1)
y = df['ConcreteCompressiveStrength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

INPUT_SHAPE = (X_train_scaled.shape[1],)


def plot_regression_history(history, title):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Training Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()


def build_and_evaluate_model(neurons, title):
    model = Sequential([
        Input(shape=INPUT_SHAPE),
        Dense(neurons, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    print(f"\n--- Training Model: {title} ---")
    history = model.fit(X_train_scaled, y_train,
                        epochs=50,
                        validation_split=0.2,
                        verbose=0)

    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Evaluation on Test Data for {title}:")
    print(f"  Mean Squared Error (MSE): {loss:.2f}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")

    plot_regression_history(history, f'Loss Over Epochs - {title}')
    return loss, mae



print("\n--- 1. Effect of Epochs ---")
epochs_to_test = [20, 50, 100]
epoch_results = {}

for num_epochs in epochs_to_test:
    model = Sequential([
        Input(shape=INPUT_SHAPE),
        Dense(BETTER_MODEL_NEURONS, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_scaled, y_train, epochs=num_epochs, verbose=0)
    loss = model.evaluate(X_test_scaled, y_test, verbose=0)
    epoch_results[num_epochs] = loss
    print(f"Test MSE with {num_epochs} epochs: {loss:.2f}")





print("\n--- 2. Comparison of Loss Functions ---")
loss_functions = {
    "MSE": "mean_squared_error",
    "MAE": "mean_absolute_error",
    "Huber": tf.keras.losses.Huber()
}
loss_results = {}

for name, loss_fn in loss_functions.items():
    model = Sequential([
        Input(shape=INPUT_SHAPE),
        Dense(BETTER_MODEL_NEURONS, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=loss_fn)
    model.fit(X_train_scaled, y_train, epochs=50, verbose=0)
    loss = model.evaluate(X_test_scaled, y_test, verbose=0)
    loss_results[name] = loss
    print(f"Test Loss ({name}) with {name} function: {loss:.2f}")




print("\n--- 3. Comparison of Optimizers ---")
optimizers = ['sgd', 'adam', 'rmsprop']
optimizer_histories = {}

for opt_name in optimizers:
    model = Sequential([
        Input(shape=INPUT_SHAPE),
        Dense(BETTER_MODEL_NEURONS, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=opt_name, loss='mean_squared_error')
    history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=0)
    optimizer_histories[opt_name] = history
    loss = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Final Test MSE with {opt_name.upper()} optimizer: {loss:.2f}")



iris = load_iris()
X = iris.data
y = iris.target


mask = (y == 0) | (y == 1)
X_subset = X[mask]
y_subset = y[mask]


X_final = X_subset[:, [2, 3]]
y_final = np.where(y_subset == 0, -1, 1)


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_final)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_final, test_size=0.3, random_state=42, stratify=y_final
)


print(f"Shape of training features (X_train): {X_train.shape}")
print(f"Shape of training labels (y_train): {y_train.shape}")
print(f"Shape of testing features (X_test): {X_test.shape}")
print(f"Shape of testing labels (y_test): {y_test.shape}")
print(f"Unique labels in training set: {np.unique(y_train)}")


class Adaline:
    def __init__(self, n_inputs, learning_rate=0.01, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand(1)[0]

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        history = {
            'weights': [],
            'biases': [],
            'errors': [],
            'accuracies': []
        }

        for epoch in range(self.epochs):
            output = self.activation(X)
            errors = y - output

            self.weights += self.learning_rate * X.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()

            cost = (errors**2).sum() / 2.0

            predictions = self.predict(X)
            accuracy = np.mean(predictions == y)

            history['weights'].append(self.weights.copy())
            history['biases'].append(self.bias)
            history['errors'].append(cost)
            history['accuracies'].append(accuracy)

        return history



learning_rates = [0.02, 0.005, 0.001]
num_epochs = 10
models_history = {}

for lr in learning_rates:
    adaline_model = Adaline(n_inputs=X_train.shape[1], learning_rate=lr, epochs=num_epochs)
    history = adaline_model.fit(X_train, y_train)
    models_history[lr] = history


for lr, history in models_history.items():
    print(f"Learning Rate {lr}:")
    print(f"  Final Error (Cost): {history['errors'][-1]:.4f}")
    print(f"  Final Accuracy: {history['accuracies'][-1]:.4f}")


def plot_decision_boundary(X, y, history, title):
    plt.figure(figsize=(10, 6))

    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='o', label='Setosa')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='x', label='Versicolor')

    w = history['weights'][-1]
    b = history['biases'][-1]

    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x2_values = [(-w[0] * x1 - b) / w[1] for x1 in [x1_min, x1_max]]

    plt.plot([x1_min, x1_max], x2_values, 'k--', lw=2, label='Decision Boundary')

    plt.title(title)
    plt.xlabel('Petal Length [normalized]')
    plt.ylabel('Petal Width [normalized]')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train_flat = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_flat = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print("--- Data Preprocessing Summary ---")
print(f"Training data shape: {x_train_flat.shape}")
print(f"Test data shape: {x_test_flat.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")


EPOCHS = 20
BATCH_SIZE = 256
INPUT_DIM = 784

latent_dim_1 = 8



encoder_input_1 = Input(shape=(INPUT_DIM,))
encoded_1 = Dense(128, activation='relu')(encoder_input_1)
encoded_1 = Dense(latent_dim_1, activation='relu')(encoded_1)
encoder_model_1 = Model(encoder_input_1, encoded_1, name="encoder_8")

decoder_input_1 = Input(shape=(latent_dim_1,))
decoded_1 = Dense(128, activation='relu')(decoder_input_1)
decoded_1 = Dense(INPUT_DIM, activation='linear')(decoded_1)
decoder_model_1 = Model(decoder_input_1, decoded_1, name="decoder_8")

autoencoder_output_1 = decoder_model_1(encoder_model_1(encoder_input_1))
autoencoder_model_1 = Model(encoder_input_1, autoencoder_output_1, name="autoencoder_8")

autoencoder_model_1.compile(optimizer='adam', loss='mse')


history_ae_1 = autoencoder_model_1.fit(
    x_train_flat, x_train_flat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test_flat, x_test_flat),
    verbose=1
)


latent_dim_2 = 4

encoder_input_2 = Input(shape=(INPUT_DIM,))
encoded_2 = Dense(128, activation='relu')(encoder_input_2)
encoded_2 = Dense(latent_dim_2, activation='relu')(encoded_2)
encoder_model_2 = Model(encoder_input_2, encoded_2, name="encoder_4")

decoder_input_2 = Input(shape=(latent_dim_2,))
decoded_2 = Dense(128, activation='relu')(decoder_input_2)
decoded_2 = Dense(INPUT_DIM, activation='linear')(decoded_2)
decoder_model_2 = Model(decoder_input_2, decoded_2, name="decoder_4")

autoencoder_output_2 = decoder_model_2(encoder_model_2(encoder_input_2))
autoencoder_model_2 = Model(encoder_input_2, autoencoder_output_2, name="autoencoder_4")

autoencoder_model_2.compile(optimizer='adam', loss='mse')


history_ae_2 = autoencoder_model_2.fit(
    x_train_flat, x_train_flat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test_flat, x_test_flat),
    verbose=1
)


encoder_model_1.trainable = False
encoder_model_2.trainable = False

NUM_CLASSES = 10


classifier_model_1 = Sequential([
    encoder_model_1,
    Dense(4, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
], name="classifier_8")

classifier_model_1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


history_cls_1 = classifier_model_1.fit(
    x_train_flat, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test_flat, y_test),
    verbose=1
)


classifier_model_2 = Sequential([
    encoder_model_2,
    Dense(NUM_CLASSES, activation='softmax')
], name="classifier_4")

classifier_model_2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


history_cls_2 = classifier_model_2.fit(
    x_train_flat, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test_flat, y_test),
    verbose=1
)


print("Autoencoder with 8-dim latent space:")
autoencoder_model_1.summary()

print("\nAutoencoder with 4-dim latent space:")
autoencoder_model_2.summary()

print("\nClassifier with 8-dim frozen encoder:")
classifier_model_1.summary()

print("\nClassifier with 4-dim frozen encoder:")
classifier_model_2.summary()


def plot_reconstructions(autoencoder, data, n=10):
    reconstructed_imgs = autoencoder.predict(data[:n])
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_imgs[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


latent_dim_b = 16


encoder_input_b = Input(shape=(INPUT_DIM,))
encoded_b = Dense(256, activation='relu')(encoder_input_b)
encoded_b = Dense(128, activation='relu')(encoded_b)
encoded_b = Dense(latent_dim_b, activation='relu')(encoded_b)
encoder_model_b = Model(encoder_input_b, encoded_b, name="encoder_bonus")

decoder_input_b = Input(shape=(latent_dim_b,))
decoded_b = Dense(128, activation='relu')(decoder_input_b)
decoded_b = Dense(256, activation='relu')(decoded_b)
decoded_b = Dense(INPUT_DIM, activation='sigmoid')(decoded_b)
decoder_model_b = Model(decoder_input_b, decoded_b, name="decoder_bonus")

autoencoder_output_b = decoder_model_b(encoder_model_b(encoder_input_b))
autoencoder_model_b = Model(encoder_input_b, autoencoder_output_b, name="autoencoder_bonus")
autoencoder_model_b.compile(optimizer='adam', loss='mse')
history_ae_b = autoencoder_model_b.fit(x_train_flat, x_train_flat, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test_flat, x_test_flat), verbose=0)


encoder_model_b.trainable = False
classifier_model_b = Sequential([
    encoder_model_b,
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
], name="classifier_bonus")
classifier_model_b.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cls_b = classifier_model_b.fit(x_train_flat, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test_flat, y_test), verbose=0)

