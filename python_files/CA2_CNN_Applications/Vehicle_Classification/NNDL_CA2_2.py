from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import numpy as np


def evaluate_classification(y_true, y_pred, class_names=None):
    """Print detailed classification metrics."""
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return {
        "accuracy": (y_pred == y_true).mean(),
        "precision": np.mean(
            [cm[i, i] / cm[:, i].sum() for i in range(len(cm)) if cm[:, i].sum() > 0]
        ),
        "recall": np.mean(
            [cm[i, i] / cm[i, :].sum() for i in range(len(cm)) if cm[i, :].sum() > 0]
        ),
        "f1": (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        ),
    }


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from torchsummary import summary

import torchvision
from torchvision import transforms, models, datasets
from torchvision.datasets import ImageFolder

import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import glob
import os
import random
import shutil


class CONFIG:
    seed = 42
    width, height = 224, 224
    path = "/content/data/toyota_image_dataset_v2/toyota_cars/"
    max_samples_per_class = 300
    samples_per_class_after_aug = 150
    num_classes = 10

    epochs = 100
    batch_size = 32
    optimizer = "adam"
    loss_function = "sparse_categorical_crossentropy"
    test_size = 0.2
    val_size = 0.05
    patience = 10
    start_from_epoch = 5


DATA_DIR = "/kaggle/input/toyota_image_dataset_v2/toyota_cars"


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

data = ImageFolder(root=DATA_DIR, transform=transform)


def show_dataset(data):
    df = pd.DataFrame(
        {
            "file_path": [sample[0] for sample in data.samples],
            "label_name": [data.classes[sample[1]] for sample in data.samples],
        }
    )

    label_counts = df["label_name"].value_counts().reset_index()
    label_counts.columns = ["label_name", "count"]

    plt.figure(figsize=(10, 5))
    sns.barplot(data=label_counts, x="label_name", y="count", palette="viridis")

    plt.xlabel("Class Labels")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution in Dataset")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


show_dataset(data)


allowed_classes = [
    "corolla",
    "camry",
    "rav4",
    "tacoma",
    "highlander",
    "prius",
    "tundra",
    "4runner",
    "yaris",
    "sienna",
]


def filter_image_folder(dataset, allowed_classes):
    filtered_indices = [
        i
        for i, (_, label) in enumerate(dataset.samples)
        if dataset.classes[label] in allowed_classes
    ]

    filtered_classes = [cls for cls in dataset.classes if cls in allowed_classes]
    filtered_class_to_idx = {cls: idx for idx, cls in enumerate(filtered_classes)}

    filtered_samples = [
        (path, filtered_class_to_idx[dataset.classes[label]])
        for path, label in dataset.samples
        if dataset.classes[label] in allowed_classes
    ]
    updated_targets = [
        filtered_class_to_idx[dataset.classes[label]]
        for label in dataset.targets
        if dataset.classes[label] in allowed_classes
    ]

    dataset.samples = filtered_samples
    dataset.targets = updated_targets
    dataset.classes = filtered_classes
    dataset.class_to_idx = filtered_class_to_idx

    return dataset


selected_data = filter_image_folder(data, allowed_classes)


print("Corrupted Images:")

corrupted = []
flag = 0
for path, _, files in os.walk(DATA_DIR):
    for f in files:
        try:
            img = Image.open(os.path.join(path, f))
            img.verify()
        except Exception as e:
            print(f"Corrupted: {f} - {e}")
            corrupted.append(os.path.join(path, f))


def remove_corrupted_images(data, corrupted):
    for corrupted_image in corrupted:
        for sample in data.samples:
            if corrupted_image == sample[0]:
                print(sample[0])
                data.samples.remove(sample)
    return data


selected_data = remove_corrupted_images(selected_data, corrupted)
show_dataset(selected_data)


n_samples = 9
indices = np.random.choice(len(selected_data), n_samples, replace=False)

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
fig.suptitle("Sample Images from the Dataset", fontsize=16)

for ax, idx in zip(axes.flatten(), indices):
    img, label = selected_data[idx]
    ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    ax.set_title(f"Class: {selected_data.classes[label]}")
    ax.axis("off")

plt.tight_layout()
plt.show()


def obtain_features(model, dataloader, device):
    model.to(device)
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for batch_samples, batch_labels in dataloader:
            batch_samples = batch_samples.to(device)

            output = model(batch_samples)

            features.append(output.cpu())
            labels.append(batch_labels)

    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataloader = DataLoader(selected_data, batch_size=128, shuffle=False)

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model.classifier = nn.Identity()

vgg_features, vgg_labels = obtain_features(model, dataloader, device)

model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
model.classifier = nn.Identity()

alexnet_features, alexnet_labels = obtain_features(model, dataloader, device)


alexnet_features.shape, vgg_features.shape


vgg_features_to_save = vgg_features.cpu().numpy()
vgg_labels_to_save = vgg_labels.cpu().numpy()
alexnet_features_to_save = alexnet_features.cpu().numpy()
alexnet_labels_to_save = alexnet_labels.cpu().numpy()
np.save("vgg_features.npy", vgg_features_to_save)
np.save("vgg_labels.npy", vgg_labels_to_save)
np.save("alexnet_features.npy", alexnet_features_to_save)
np.save("alexnet_labels.npy", alexnet_labels_to_save)


torch.equal(vgg_labels, alexnet_labels)


vgg_features_train, vgg_features_test, vgg_labels_train, vgg_labels_test = (
    train_test_split(vgg_features, vgg_labels, test_size=0.2, random_state=42)
)
(
    alexnet_features_train,
    alexnet_features_test,
    alexnet_labels_train,
    alexnet_labels_test,
) = train_test_split(alexnet_features, alexnet_labels, test_size=0.2, random_state=42)


classification_result = {}


def plot_confusion_matrix(
    cm, class_names=None, figsize=(8, 6), title="Confusion Matrix"
):
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def split_train_val_test(dataset, first_ratio, second_ratio):
    generator = torch.Generator().manual_seed(42)
    first_size = int(first_ratio * len(dataset))
    if first_ratio + second_ratio == 1:
        second_size = len(dataset) - first_size
        first_partition, second_partition = random_split(
            dataset, [first_size, second_size], generator=generator
        )
        return first_partition, second_partition, None
    else:
        second_size = int(second_ratio * len(dataset))
        third_size = len(dataset) - first_size - second_size
        return random_split(
            dataset, [first_size, second_size, third_size], generator=generator
        )


def fit(model, optimizer, criterion, train_loader, device):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        counter += 1
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    train_accuracy = 100.0 * train_running_correct / total
    return train_loss, train_accuracy


def validation(model, val_loader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    for data, target in val_loader:
        counter += 1
        data, target = data.to(device), target.to(device)
        total += target.size(0)
        outputs = model(data)
        loss = criterion(outputs, target)
        val_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        val_running_correct += (preds == target).sum().item()

    val_loss = val_running_loss / counter
    val_accuracy = 100.0 * val_running_correct / total
    return val_loss, val_accuracy


def plot_history(train_hist, val_hist, name):
    plt.plot(train_hist)
    plt.plot(val_hist)
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.legend(["train", "val"])
    plt.show()


def train(model, train_loader, val_loader, optimizer, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")
        train_epoch_loss, train_epoch_accuracy = fit(
            model, optimizer, criterion, train_loader, device
        )
        val_epoch_loss, val_epoch_accuracy = validation(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f},\
        Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}"
        )

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

    plot_history(train_losses, val_losses, "loss")
    plot_history(train_accuracies, val_accuracies, "accuracy")
    return model


def evaluate_model(model, data_loader, device="cuda"):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    precision = precision_score(all_labels, all_predictions, average="weighted")

    cm = confusion_matrix(all_labels, all_predictions)

    classification_result = {
        "Accuracy": accuracy,
        "f1-score": f1,
        "Recall": recall,
        "Precision": precision,
    }

    return classification_result, np.array(cm)


class VGG16_classifier(nn.Module):
    def __init__(self, output_dim=10):
        super(VGG16_classifier, self).__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        self.fc_layers = nn.Sequential(
            *list(vgg.classifier.children())[:-1], nn.Linear(4096, output_dim)
        )

    def forward(self, x):
        return self.fc_layers(x)


model = VGG16_classifier()
model.to(device)
model.eval()


summary(model, input_size=(1, 25088))


X_train_tensor = vgg_features_train.float()
y_train_tensor = vgg_labels_train.long()

X_test_tensor = vgg_features_test.float()
y_test_tensor = vgg_labels_test.long()

vgg_tmp_dataset = TensorDataset(X_train_tensor, y_train_tensor)
vgg_test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

vgg_train_dataset, vgg_val_dataset, _ = split_train_val_test(
    vgg_tmp_dataset, 0.95, 0.05
)

vgg_train_loader = DataLoader(vgg_train_dataset, batch_size=32, shuffle=True)
vgg_val_loader = DataLoader(vgg_val_dataset, batch_size=32, shuffle=True)
vgg_test_loader = DataLoader(vgg_test_dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, vgg_train_loader, vgg_val_loader, optimizer, 15, device)

classification_result["VGG16"], vgg_cm = evaluate_model(model, vgg_test_loader, device)


print(classification_result["VGG16"])


class AlexNet_classifier(nn.Module):
    def __init__(self, output_dim=10):
        super(AlexNet_classifier, self).__init__()

        alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

        self.fc_layers = nn.Sequential(
            *list(alexnet.classifier.children())[:-1], nn.Linear(4096, output_dim)
        )

    def forward(self, x):
        return self.fc_layers(x)


model = AlexNet_classifier()
model.to(device)
model.eval()


summary(model, (1, 9216))


X_train_tensor = alexnet_features_train.float()
y_train_tensor = alexnet_labels_train.long()

X_test_tensor = alexnet_features_test.float()
y_test_tensor = alexnet_labels_test.long()

alexnet_tmp_dataset = TensorDataset(X_train_tensor, y_train_tensor)
alexnet_test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

alexnet_train_dataset, alexnet_val_dataset, _ = split_train_val_test(
    alexnet_tmp_dataset, 0.95, 0.05
)

alexnet_train_loader = DataLoader(alexnet_train_dataset, batch_size=32, shuffle=True)
alexnet_val_loader = DataLoader(alexnet_val_dataset, batch_size=32, shuffle=True)
alexnet_test_loader = DataLoader(alexnet_test_dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, alexnet_train_loader, alexnet_val_loader, optimizer, 15, device)

classification_result["AlexNet"], alexnet_cm = evaluate_model(
    model, alexnet_test_loader, device
)


print(classification_result["AlexNet"])


class CNN_CONFIG:
    input_dimension = (224, 224, 3)
    filter_to_learn = (64, 64, 128, 128, 256, 256)
    max_pooling = (2, 2)
    cnn_activation_function = "relu"
    fcn_number_of_neurons = (512, 256)
    fcn_activation_function = ("relu", "relu")
    fcn_output_activation = "softmax"
    dropout_rate = 0.2
    kernel_size = (3, 3)
    number_of_cnn_layers = 6
    number_of_fcn_layers = 3
    num_classes = 10


class ToyotaModelCNN(nn.Module):
    def __init__(self, config):
        super(ToyotaModelCNN, self).__init__()
        self.config = config

        layers = []
        in_channels = config.input_dimension[2]

        for i in range(config.number_of_cnn_layers):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    config.filter_to_learn[i],
                    kernel_size=config.kernel_size,
                    padding=1,
                )
            )
            layers.append(nn.MaxPool2d(kernel_size=config.max_pooling))
            layers.append(nn.BatchNorm2d(config.filter_to_learn[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            in_channels = config.filter_to_learn[i]

        self.cnn = nn.Sequential(*layers)

        dummy_input = torch.zeros(
            1,
            config.input_dimension[2],
            config.input_dimension[0],
            config.input_dimension[1],
        )
        with torch.no_grad():
            dummy_output = self.cnn(dummy_input)
        flattened_size = dummy_output.view(1, -1).shape[1]

        fcn_layers = [nn.Flatten()]
        in_features = flattened_size
        for i in range(config.number_of_fcn_layers - 1):
            fcn_layers.append(nn.Linear(in_features, config.fcn_number_of_neurons[i]))
            fcn_layers.append(nn.BatchNorm1d(config.fcn_number_of_neurons[i]))
            fcn_layers.append(nn.ReLU())
            in_features = config.fcn_number_of_neurons[i]

        fcn_layers.append(nn.Linear(in_features, config.num_classes))
        self.fcn = nn.Sequential(*fcn_layers)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fcn(x)
        if self.config.fcn_output_activation == "softmax":
            x = F.softmax(x, dim=1)
        return x


model = ToyotaModelCNN(CNN_CONFIG())
model.to(device)
model.eval()


summary(model, (3, 224, 224))


cnn_train_dataset, cnn_test_dataset, _ = split_train_val_test(selected_data, 0.8, 0.2)
cnn_train_dataset, cnn_val_dataset, _ = split_train_val_test(
    cnn_train_dataset, 0.95, 0.05
)

cnn_train_loader = DataLoader(
    cnn_train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
cnn_val_loader = DataLoader(
    cnn_val_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
cnn_test_loader = DataLoader(
    cnn_test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, cnn_train_loader, cnn_val_loader, optimizer, 15, device)

classification_result["CNN"], cnn_cm = evaluate_model(model, cnn_test_loader, device)


print(classification_result["CNN"])


with torch.no_grad():
    flat_features_train = vgg_features_train.view(vgg_features_train.size(0), -1)
    flat_features_test = vgg_features_test.view(vgg_features_test.size(0), -1)
svm_features_train = flat_features_train.cpu().numpy()
svm_features_test = flat_features_test.cpu().numpy()

svm_labels_train = vgg_labels_train.cpu().numpy()
svm_labels_test = vgg_labels_test.cpu().numpy()


scaler = StandardScaler()
svm_features_train = scaler.fit_transform(svm_features_train)
svm_features_test = scaler.transform(svm_features_test)

svm = SVC(C=1.0, kernel="linear")
svm.fit(svm_features_train, svm_labels_train)

predictions = svm.predict(svm_features_test)
classification_result["VGG+SVM"] = {
    "Accuracy": accuracy_score(svm_labels_test, predictions),
    "f1-score": f1_score(svm_labels_test, predictions, average="weighted"),
    "Recall": recall_score(svm_labels_test, predictions, average="weighted"),
    "Precision": precision_score(svm_labels_test, predictions, average="weighted"),
}
vgg_svm_cm = confusion_matrix(svm_labels_test, predictions)


print(classification_result["VGG+SVM"])


def compare_models(classification_result, model_names):
    accuracies = []
    precisions = []
    recalls = []
    f_scores = []
    for name in model_names:
        accuracies.append(classification_result[name]["Accuracy"] * 100)
        precisions.append(classification_result[name]["Precision"] * 100)
        recalls.append(classification_result[name]["Recall"] * 100)
        f_scores.append(classification_result[name]["f1-score"] * 100)

    x = np.arange(4)
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.figure(figsize=(3 * len(model_names), 5))
    colors = ["dodgerblue", "indianred", "yellowgreen", "mediumpurple"]
    for i in range(len(model_names)):
        ax.bar(
            x + width * i,
            [accuracies[i], precisions[i], recalls[i], f_scores[i]],
            color=colors[i % len(colors)],
            width=width,
            label=model_names[i],
        )
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1 Score"])
    ax.yaxis.grid(True, color="gray")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=len(model_names), bbox_to_anchor=(0.5, -0.1))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(cms, model_names, classes):
    num_models = len(model_names)
    rows = 2
    cols = (num_models + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= num_models:
            ax.axis("off")
            break
        matrix = cms[i]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar=True,
            ax=ax,
        )
        ax.text(
            0.5,
            -0.3,
            model_names[i],
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="top",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Prediction", fontsize=8)
        ax.set_ylabel("Ground Truth", fontsize=8)
    plt.tight_layout(h_pad=3.0)
    plt.show()


model_names = ["VGG+SVM", "VGG16", "AlexNet", "CNN"]


compare_models(classification_result, model_names)


classification_result


plot_confusion_matrices(
    [vgg_cm, alexnet_cm, cnn_cm, vgg_svm_cm], model_names, selected_data.classes
)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


dataset = ImageFolder(root=DATA_DIR, transform=transform)


random.seed(42)

BASE_OUTPUT_DIR = "splitted_dataset"
TRAIN_DIR = os.path.join(BASE_OUTPUT_DIR, "train")
TEST_DIR = os.path.join(BASE_OUTPUT_DIR, "test")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

split_ratio = 0.8

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    all_files = os.listdir(class_path)

    images = [
        fname for fname in all_files if os.path.splitext(fname)[1].lower() in valid_exts
    ]

    unknown_exts = [
        fname
        for fname in all_files
        if os.path.splitext(fname)[1].lower() not in valid_exts
    ]
    if unknown_exts:
        print(f"[Warning] Unexpected files found in '{class_name}': {unknown_exts}")

    if not images:
        continue

    random.shuffle(images)
    split_point = int(len(images) * split_ratio)
    train_images = images[:split_point]
    test_images = images[split_point:]

    for subset, file_list in [("train", train_images), ("test", test_images)]:
        subset_dir = os.path.join(BASE_OUTPUT_DIR, subset, class_name)
        os.makedirs(subset_dir, exist_ok=True)

        for img in file_list:
            src = os.path.join(class_path, img)
            dst = os.path.join(subset_dir, img)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            else:
                print(f"[Warning] File not found: {src}")

print("Dataset split completed.")


print("Corrupted Train Images:")

corrupted_train = []
flag = 0
for path, _, files in os.walk(TRAIN_DIR):
    for f in files:
        try:
            img = Image.open(os.path.join(path, f))
            img.verify()
        except Exception as e:
            print(f"Corrupted: {f} - {e}")
            corrupted_train.append(os.path.join(path, f))

print("Corrupted Test Images:")

corrupted_test = []
flag = 0
for path, _, files in os.walk(TEST_DIR):
    for f in files:
        try:
            img = Image.open(os.path.join(path, f))
            img.verify()
        except Exception as e:
            print(f"Corrupted: {f} - {e}")
            corrupted_test.append(os.path.join(path, f))


allowed_classes = [
    "corolla",
    "camry",
    "rav4",
    "tacoma",
    "highlander",
    "prius",
    "tundra",
    "4runner",
    "yaris",
    "sienna",
]
selected_train_data = filter_image_folder(
    ImageFolder(root=TRAIN_DIR, transform=transform), allowed_classes
)
selected_train_data = remove_corrupted_images(selected_train_data, corrupted_train)
selected_test_data = filter_image_folder(
    ImageFolder(root=TEST_DIR, transform=transform), allowed_classes
)
selected_test_data = remove_corrupted_images(selected_test_data, corrupted_test)


class BalancedImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        min_samples=2000,
        augmentations=None,
        allowed_classes=None,
    ):
        super().__init__(root, transform)

        if allowed_classes:
            filter_image_folder(self, allowed_classes)

        self.min_samples = min_samples
        self.augmentations = (
            augmentations if augmentations else self.default_augmentations()
        )

        self.samples, self.targets, self.flags = self.balance_classes()
        self.imgs = self.samples

    def default_augmentations(self):
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomApply(
                    [transforms.Grayscale(num_output_channels=3)], p=0.3
                ),
            ]
        )

    def balance_classes(self):

        samples_by_class = {class_idx: [] for class_idx in range(len(self.classes))}
        for sample in self.samples:
            samples_by_class[sample[1]].append(sample)

        new_samples = []
        new_targets = []
        new_flags = []

        for class_idx, sample_list in samples_by_class.items():
            for sample in sample_list:
                new_samples.append(sample)
                new_targets.append(class_idx)
                new_flags.append(False)

            count = len(sample_list)
            if count < self.min_samples:
                needed = self.min_samples - count
                for _ in range(needed):
                    extra_sample = random.choice(sample_list)
                    new_samples.append(extra_sample)
                    new_targets.append(class_idx)
                    new_flags.append(True)

        return new_samples, new_targets, new_flags

    def __getitem__(self, index):

        img_path = self.samples[index][0]
        label = self.targets[index]
        is_augmented = self.flags[index]

        img = Image.open(img_path).convert("RGB")

        if is_augmented:
            img = self.augmentations(img)

        if self.transform:
            img = self.transform(img)

        return img, label


balanced_train_data = filter_image_folder(
    BalancedImageFolder(
        root=TRAIN_DIR, transform=transform, allowed_classes=allowed_classes
    ),
    allowed_classes,
)
balanced_train_data = remove_corrupted_images(balanced_train_data, corrupted_train)


def default_augmentations():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomApply(
                [transforms.Grayscale(num_output_channels=3)], p=0.3
            ),
        ]
    )


plt.figure(figsize=(10, 10))
ind = np.random.choice(np.arange(len(balanced_train_data)))
img, _ = balanced_train_data[ind]
plt.subplot(3, 3, 1)
plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
plt.title("Before augmentaion")
plt.axis("off")
for i in range(1, 6):
    plt.subplot(3, 3, i + 1)
    plt.imshow(np.transpose(default_augmentations()(img).numpy(), (1, 2, 0)))
    plt.title("After augmentaion")
    plt.axis("off")
plt.show()


show_dataset(balanced_train_data)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader = DataLoader(balanced_train_data, batch_size=128, shuffle=False)
test_loader = DataLoader(selected_test_data, batch_size=128, shuffle=False)


model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model.classifier = nn.Identity()

vgg_features_train, vgg_labels_train = obtain_features(model, train_loader, device)
vgg_features_test, vgg_labels_test = obtain_features(model, test_loader, device)


model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
model.classifier = nn.Identity()

alexnet_features_train, alexnet_labels_train = obtain_features(
    model, train_loader, device
)
alexnet_features_test, alexnet_labels_test = obtain_features(model, test_loader, device)


vgg_features_to_save = vgg_features_train.cpu().numpy()
vgg_labels_to_save = vgg_labels_train.cpu().numpy()
alexnet_features_to_save = alexnet_features_train.cpu().numpy()
alexnet_labels_to_save = alexnet_labels_train.cpu().numpy()
np.save("vgg_features_train.npy", vgg_features_to_save)
np.save("vgg_labels_train.npy", vgg_labels_to_save)
np.save("alexnet_features_train.npy", alexnet_features_to_save)
np.save("alexnet_labels_train.npy", alexnet_labels_to_save)


vgg_features_to_save = vgg_features_test.cpu().numpy()
vgg_labels_to_save = vgg_labels_test.cpu().numpy()
alexnet_features_to_save = alexnet_features_test.cpu().numpy()
alexnet_labels_to_save = alexnet_labels_test.cpu().numpy()
np.save("vgg_features_test.npy", vgg_features_to_save)
np.save("vgg_labels_test.npy", vgg_labels_to_save)
np.save("alexnet_features_test.npy", alexnet_features_to_save)
np.save("alexnet_labels_test.npy", alexnet_labels_to_save)


classification_result = {}


model = VGG16_classifier()
model.to(device)


X_train_tensor = vgg_features_train.float()
y_train_tensor = vgg_labels_train.long()

X_test_tensor = vgg_features_test.float()
y_test_tensor = vgg_labels_test.long()

vgg_tmp_dataset = TensorDataset(X_train_tensor, y_train_tensor)
vgg_test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

vgg_train_dataset, vgg_val_dataset, _ = split_train_val_test(
    vgg_tmp_dataset, 0.95, 0.05
)

vgg_train_loader = DataLoader(vgg_train_dataset, batch_size=32, shuffle=True)
vgg_val_loader = DataLoader(vgg_val_dataset, batch_size=32, shuffle=True)
vgg_test_loader = DataLoader(vgg_test_dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, vgg_train_loader, vgg_val_loader, optimizer, 15, device)

classification_result["VGG16"], vgg_cm = evaluate_model(model, vgg_test_loader, device)


print(classification_result["VGG16"])


model = AlexNet_classifier()
model.to(device)


X_train_tensor = alexnet_features_train.float()
y_train_tensor = alexnet_labels_train.long()

X_test_tensor = alexnet_features_test.float()
y_test_tensor = alexnet_labels_test.long()

alexnet_tmp_dataset = TensorDataset(X_train_tensor, y_train_tensor)
alexnet_test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

alexnet_train_dataset, alexnet_val_dataset, _ = split_train_val_test(
    alexnet_tmp_dataset, 0.95, 0.05
)

alexnet_train_loader = DataLoader(alexnet_train_dataset, batch_size=32, shuffle=True)
alexnet_val_loader = DataLoader(alexnet_val_dataset, batch_size=32, shuffle=True)
alexnet_test_loader = DataLoader(alexnet_test_dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, alexnet_train_loader, alexnet_val_loader, optimizer, 15, device)

classification_result["AlexNet"], alexnet_cm = evaluate_model(
    model, alexnet_test_loader, device
)


print(classification_result["AlexNet"])


model = ToyotaModelCNN(CNN_CONFIG())
model.to(device)


cnn_train_dataset, cnn_val_dataset, _ = split_train_val_test(
    balanced_train_data, 0.95, 0.05
)

cnn_train_loader = DataLoader(
    cnn_train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
cnn_val_loader = DataLoader(
    cnn_val_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
cnn_test_loader = DataLoader(
    selected_test_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, cnn_train_loader, cnn_val_loader, optimizer, 15, device)

classification_result["CNN"], cnn_cm = evaluate_model(model, cnn_test_loader, device)


print(classification_result["CNN"])


with torch.no_grad():
    flat_features_train = vgg_features_train.view(vgg_features_train.size(0), -1)
    flat_features_test = vgg_features_test.view(vgg_features_test.size(0), -1)
svm_features_train = flat_features_train.cpu().numpy()
svm_features_test = flat_features_test.cpu().numpy()

svm_labels_train = vgg_labels_train.cpu().numpy()
svm_labels_test = vgg_labels_test.cpu().numpy()

scaler = StandardScaler()
svm_features_train = scaler.fit_transform(svm_features_train)
svm_features_test = scaler.transform(svm_features_test)

svm = SVC(C=50, kernel="linear")
svm.fit(svm_features_train, svm_labels_train)

predictions = svm.predict(svm_features_test)
classification_result["VGG+SVM"] = {
    "Accuracy": accuracy_score(svm_labels_test, predictions),
    "f1-score": f1_score(svm_labels_test, predictions, average="weighted"),
    "Recall": recall_score(svm_labels_test, predictions, average="weighted"),
    "Precision": precision_score(svm_labels_test, predictions, average="weighted"),
}
print(f"Accuracy: {classification_result['VGG+SVM']['Accuracy']}")
print(f"f1-score: {classification_result['VGG+SVM']['f1-score']}")


model_names = ["VGG+SVM", "VGG16", "AlexNet", "CNN"]


compare_models(classification_result, model_names)


classification_result


plot_confusion_matrices(
    [vgg_cm, alexnet_cm, cnn_cm, vgg_svm_cm], model_names, selected_data.classes
)


svm = SVC(C=50, kernel="rbf")
svm.fit(svm_features_train, svm_labels_train)

predictions = svm.predict(svm_features_test)
classification_result["VGG+SVM_RBF"] = {
    "Accuracy": accuracy_score(svm_labels_test, predictions),
    "f1-score": f1_score(svm_labels_test, predictions, average="weighted"),
    "Recall": recall_score(svm_labels_test, predictions, average="weighted"),
    "Precision": precision_score(svm_labels_test, predictions, average="weighted"),
}
print(f"Accuracy: {classification_result['VGG+SVM_RBF']['Accuracy']}")
print(f"f1-score: {classification_result['VGG+SVM_RBF']['f1-score']}")
