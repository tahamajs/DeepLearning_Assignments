from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize


def evaluate_medical_classification(
    y_true, y_pred, y_pred_proba=None, class_names=None
):
    """Comprehensive evaluation for medical image classification."""
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix (Medical Classification)")
    plt.tight_layout()
    plt.show()

    if y_pred_proba is not None and len(np.unique(y_true)) > 2:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_true_bin.shape[1]

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auc_score = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc_score:.3f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-class ROC Curves")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "accuracy": (y_pred == y_true).mean(),
        "macro_precision": np.mean(
            [cm[i, i] / cm[:, i].sum() for i in range(len(cm)) if cm[:, i].sum() > 0]
        ),
        "macro_recall": np.mean(
            [cm[i, i] / cm[i, :].sum() for i in range(len(cm)) if cm[i, :].sum() > 0]
        ),
        "macro_f1": np.mean(
            [
                2 * cm[i, i] / (cm[i, :].sum() + cm[:, i].sum())
                for i in range(len(cm))
                if (cm[i, :].sum() + cm[:, i].sum()) > 0
            ]
        ),
    }


from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import io
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionV3
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.metrics import precision_recall_fscore_support as score
from collections import defaultdict
from tensorflow.keras import ops


ds = load_dataset("ArianFiroozi/NNDL_HW5_S2025")


ds


df = ds["train"].to_pandas()
df.info()


samples = df.groupby("label").sample(n=1)
n_classes = len(samples)
plt.figure(figsize=(20, 10))
for i, sample in samples.reset_index(drop=True).iterrows():
    plt.subplot(2, (n_classes + 1) // 2, i + 1)
    img = Image.open(io.BytesIO(sample["image"]["bytes"]))
    plt.imshow(img)
    plt.title(sample["disease_name"])
    plt.axis("off")


np.array(img).shape


label_counts = df[["label", "disease_name"]].value_counts().reset_index()
classes = label_counts.sort_values("label")["disease_name"]
label_counts


label_to_name = {
    row["label"]: row["disease_name"] for i, row in label_counts.iterrows()
}
name_to_label = {
    row["disease_name"]: row["label"] for i, row in label_counts.iterrows()
}


plt.figure(figsize=(10, 8))
sns.barplot(
    data=label_counts,
    x="disease_name",
    y="count",
    palette="viridis",
)

plt.xlabel("Class Labels")
plt.ylabel("Number of Images")
plt.title("Class Distribution in Dataset")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


def split_data(df, shape, num_test_per_class):
    def decode_bytes_to_np(img_bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(shape)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    df["np_image"] = df["image"].apply(lambda x: decode_bytes_to_np(x["bytes"]))

    test_df = df.groupby("label").sample(n=num_test_per_class, random_state=42)
    X_test, y_test = test_df["np_image"], test_df["label"]

    train_df = df.drop(test_df.index)
    X_train = train_df["np_image"]
    y_train = train_df["label"]

    X_train = np.stack(X_train)
    X_test = np.stack(X_test)
    y_train = np.stack(y_train)
    y_test = np.stack(y_test)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_data(df, (75, 75), 100)
X_train.shape, X_test.shape


def create_aug_layers(input_size, mean=[0, 0, 0], var=[1, 1, 1]):
    transforms = tf.keras.models.Sequential(
        [
            layers.Input(shape=(input_size, input_size, 3)),
            layers.RandomBrightness(0.1, (0.0, 1.0)),
            layers.RandomRotation(0.125, fill_mode="reflect"),
            layers.RandomZoom((-0.05, 0.05), (-0.05, 0.05), fill_mode="reflect"),
            layers.RandomFlip("horizontal"),
            layers.Normalization(mean=mean, variance=var, axis=-1),
        ],
        name="Augmentation_Layers",
    )
    return transforms


plt.figure(figsize=(10, 10))
idx = np.random.randint(0, len(X_train) + 1, 1)[0]
image = X_train[idx]
plt.subplot(3, 3, 1)
plt.imshow(image)
plt.title("Before augmentaion")
plt.axis("off")
for i in range(1, 6):
    plt.subplot(3, 3, i + 1)
    plt.imshow(tf.squeeze(create_aug_layers(75)(tf.expand_dims(image, axis=0))))
    plt.title("After augmentaion")
    plt.axis("off")
plt.show()


def oversample_images(images, labels, target_label, num_per_class):
    augmented_images = images
    mask = labels == target_label
    mask_images = images[mask]

    generated = len(mask_images)
    augmented_labels = np.concatenate(
        (labels, np.ones(num_per_class - generated) * target_label)
    )

    while generated < num_per_class:
        remaining = num_per_class - generated
        images = mask_images
        if len(images) > remaining:
            indices = np.random.choice(len(images), remaining, replace=False)
            images = mask_images[indices]

        generated += len(images)
        augmented_images = np.concatenate((augmented_images, images))

    indices = np.arange(len(augmented_labels))
    np.random.shuffle(indices)
    augmented_images = np.array(augmented_images)[indices]
    augmented_labels = np.array(augmented_labels, dtype=np.uint8)[indices]
    return augmented_images, augmented_labels


input_size = 75
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
epochs = 30
num_classes = 10
loss_fn = "sparse_categorical_crossentropy"


target_label = 9
num_per_class = (y_train == 0).sum()
X_train, y_train = oversample_images(X_train, y_train, target_label, num_per_class)


target_label = 5
X_train, y_train = oversample_images(X_train, y_train, target_label, num_per_class)


X_train.shape


mean = X_train.mean(axis=(0, 1, 2))
var = X_train.var(axis=(0, 1, 2))
aug_layers = create_aug_layers(input_size, mean, var)


def create_cnn_model(input_size, num_classes, aug_layers):
    inception = InceptionV3(
        include_top=False,
        weights=None,
        input_shape=(input_size, input_size, 3),
        pooling="avg",
    )
    model = tf.keras.models.Sequential()
    model.add(aug_layers)
    model.add(inception)
    model.add(layers.Dense(units=num_classes, activation="softmax"))
    return model


cnn = create_cnn_model(input_size, num_classes, aug_layers)
cnn.summary()


cnn = create_cnn_model(input_size, num_classes, aug_layers)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, weight_decay=weight_decay
)
cnn.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
cnn_hist = cnn.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    verbose=2,
)


def plot_history(history, model_name):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{model_name} Model Accuracy over Epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"{model_name} Model Loss over Epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


plot_history(cnn_hist, "Inception V3")


def evaluate(predicted_proba, actual_values):
    predictions = np.argmax(predicted_proba, axis=1)
    scores = defaultdict(lambda: defaultdict(float))
    precision, recall, fscore, support = score(actual_values, predictions)

    def update_scores(metric, name):
        for i, c in enumerate(classes):
            scores[name][c] = metric[i]

    metric_name = "Accuracy"
    scores[metric_name]["Micro"] = accuracy_score(actual_values, predictions)
    scores[metric_name]["Macro"] = accuracy_score(actual_values, predictions)
    matrix = confusion_matrix(actual_values, predictions)
    update_scores(matrix.diagonal() / matrix.sum(axis=1), metric_name)

    metric_name = "Precision"
    update_scores(precision, metric_name)
    scores[metric_name]["Micro"] = precision_score(
        actual_values, predictions, average="micro"
    )
    scores[metric_name]["Macro"] = precision_score(
        actual_values, predictions, average="macro"
    )

    metric_name = "Recall"
    update_scores(recall, metric_name)
    scores[metric_name]["Micro"] = precision_score(
        actual_values, predictions, average="micro"
    )
    scores[metric_name]["Macro"] = precision_score(
        actual_values, predictions, average="macro"
    )

    metric_name = "F1 score"
    update_scores(fscore, metric_name)
    scores[metric_name]["Micro"] = f1_score(actual_values, predictions, average="micro")
    scores[metric_name]["Macro"] = f1_score(actual_values, predictions, average="macro")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix, annot=True, cmap="RdYlGn", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()

    return pd.DataFrame(scores)


preds = cnn.predict(X_test)
scores = evaluate(preds, y_test)


scores.round(4)


input_size = 64
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
transformer_dim = 2
projection_layers = 8
num_classes = 10
embedding_dim = 64
epochs = 30
loss_fn = "sparse_categorical_crossentropy"


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches


plt.figure(figsize=(4, 4))
idx = np.random.choice(range(df.shape[0]), 1)
img_bytes = df["image"][idx[0]]["bytes"]
image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
image = np.asarray(image, dtype=np.float32) / 255.0
plt.imshow(image)
plt.axis("off")

resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(input_size, input_size)
)
print(f"Image size: {input_size} X {input_size}")
for patch_size in [4, 6, 8]:
    patches = Patches(patch_size)(resized_image)
    print(f"Patches per image: {patches.shape[1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    plt.suptitle(f"Patches with patch size {patch_size}x{patch_size}", fontsize=12)
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(ops.convert_to_numpy(patch_img))
        plt.axis("off")
    plt.show()


patch_size = 6


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded


class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, num_units1, num_units2):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.norm1 = layers.LayerNormalization()
        self.dense1 = layers.Dense(num_units1, activation="relu")
        self.dropout1 = layers.Dropout(0.1)
        self.dense2 = layers.Dense(num_units2)
        self.dropout2 = layers.Dropout(0.1)
        self.norm2 = layers.LayerNormalization()

    def call(self, inputs, training=False):
        attended = self.attn(query=inputs, value=inputs, key=inputs)
        x0 = self.norm1(attended + inputs)
        x = self.dropout1(x0, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.norm2(x + x0)
        return x


class TransformerBlockModel(tf.keras.Model):
    def __init__(self, num_heads, key_dim, num_units1, num_units2):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.norm1 = layers.LayerNormalization()
        self.dense1 = layers.Dense(num_units1, activation="relu")
        self.dropout1 = layers.Dropout(0.1)
        self.dense2 = layers.Dense(num_units2)
        self.dropout2 = layers.Dropout(0.1)
        self.norm2 = layers.LayerNormalization()

    def call(self, inputs, training=False):
        attended = self.attn(query=inputs, value=inputs, key=inputs)
        x0 = self.norm1(attended + inputs)
        x = self.dropout1(x0, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.norm2(x + x0)
        return x


transformer = TransformerBlockModel(transformer_dim, embedding_dim, 128, 64)
input_array = np.random.rand(256, 100, 64)
transformer(input_array)
transformer.summary()


def create_vit_classifier(
    input_shape,
    patch_size,
    embedding_dim,
    projection_layers,
    transformer_dim,
    num_classes,
    aug_layers,
):
    inputs = tf.keras.Input(shape=input_shape)
    augmented = aug_layers(inputs)
    patches = Patches(patch_size)(augmented)
    x = PatchEncoder(patches.shape[1], embedding_dim)(patches)
    for _ in range(projection_layers):
        x = TransformerBlock(transformer_dim, embedding_dim, 128, 64)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(2048, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    logits = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


X_train, X_test, y_train, y_test = split_data(df, (input_size, input_size), 100)
X_train.shape, X_test.shape


target_label = 9
num_per_class = (y_train == 0).sum()
X_train, y_train = oversample_images(X_train, y_train, target_label, num_per_class)

target_label = 5
X_train, y_train = oversample_images(X_train, y_train, target_label, num_per_class)

X_train.shape


mean = X_train.mean(axis=(0, 1, 2))
var = X_train.var(axis=(0, 1, 2))
aug_layers = create_aug_layers(input_size, mean, var)


vit = create_vit_classifier(
    (input_size, input_size, 3),
    patch_size,
    embedding_dim,
    projection_layers,
    transformer_dim,
    num_classes,
    aug_layers,
)
vit.summary()


vit = create_vit_classifier(
    (input_size, input_size, 3),
    patch_size,
    embedding_dim,
    projection_layers,
    transformer_dim,
    num_classes,
    aug_layers,
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, weight_decay=weight_decay
)
vit.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
vit_hist = vit.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    verbose=2,
)


plot_history(vit_hist, "ViT")


preds = vit.predict(X_test)
scores = evaluate(preds, y_test)


scores.round(4)
