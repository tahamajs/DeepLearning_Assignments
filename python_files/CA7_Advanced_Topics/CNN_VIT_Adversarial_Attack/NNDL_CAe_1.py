get_ipython().system("pip install torchinfo grad-cam")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from torchinfo import summary
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import itertools
from collections import defaultdict
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms as transforms
from torch.utils.data import Subset, random_split, DataLoader, Dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import time

np.random.seed(42)


from google.colab import drive

drive.mount("/content/drive/")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cifar100_class_names = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


class CustomDataset(Dataset):
    """
    Custom dataset class that allows dynamic setting of transforms.

    This is useful for applying different augmentations or normalizations
    at different stages of training (e.g., clean vs. adversarial).

    Attributes:
        dataset: The underlying torchvision dataset.
        transform: The current transform to apply.
    """

    def __init__(self, dataset, transform=None):
        """
        Initialize the custom dataset.

        Args:
            dataset: The base dataset (e.g., CIFAR100 or Flowers102).
            transform: Initial transform to apply.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)

    def set_transform(self, transform):
        """
        Dynamically set the transform.

        Args:
            transform: The new transform to apply.
        """
        self.transform = transform

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index: Index of the item.

        Returns:
            tuple: (transformed_image, label)
        """
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label


class AdversarialDataLoader:
    """
    DataLoader wrapper for adversarial examples.
    """

    def __init__(
        self,
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        attack=None,
        mean=None,
        std=None,
        device="cpu",
        attack_args=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.attack = attack
        self.mean = mean
        self.std = std
        self.device = device
        self.attack_args = attack_args or {}
        self.mode = "clean"

    def set_clean_mode(self):
        self.mode = "clean"

    def set_attack_mode(self):
        self.mode = "attack"

    def set_both_mode(self):
        self.mode = "both"

    def __iter__(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.mode == "clean":
                yield images, labels
            elif self.mode == "attack":
                adv_images = self.attack(images, labels, **self.attack_args)
                yield adv_images, labels
            elif self.mode == "both":
                adv_images = self.attack(images, labels, **self.attack_args)
                yield images, adv_images, labels

    def __len__(self):
        return len(self.dataset) // self.batch_size


num_train = 10000
train_size = int(0.8 * num_train)
val_size = num_train - train_size

restrainvalset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True)
indices = np.random.choice(len(restrainvalset), num_train)
restrainvalset = Subset(restrainvalset, indices)

len_trainval = len(restrainvalset)
restrainset, resvalset = random_split(restrainvalset, [train_size, val_size])

restrainset = CustomDataset(restrainset)
resvalset = CustomDataset(resvalset)
restrainvalset = CustomDataset(restrainvalset)

restestset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True)
restestset = CustomDataset(restestset)

len(restrainset), len(resvalset), len(restestset)


def add_noise(image, mean, variance):
    noise = np.random.normal(mean, np.sqrt(variance), image.shape)
    return image + noise


n_sample = 5
indices = np.random.choice(len(restrainset), n_sample)

plt.figure(figsize=(10, 5))
for i in range(n_sample):
    img, label = restrainset[indices[i]]
    img = np.array(img) / 255.0
    plt.subplot(2, n_sample, i + 1)
    plt.imshow(img)
    plt.title(cifar100_class_names[label])
    plt.axis("off")

    noisy_img = add_noise(img, 0, 0.05)
    plt.subplot(2, n_sample, i + 1 + n_sample)
    plt.imshow(noisy_img.clip(0, 1))
    plt.title(f"noisy\n" + cifar100_class_names[label])
    plt.axis("off")


cifar_mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
cifar_std = torch.tensor([0.2673, 0.2564, 0.2762]).view(3, 1, 1)

cifar_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(cifar_mean, cifar_std),
    ]
)

noisy_cifar_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        v2.GaussianNoise(mean=0.0, sigma=np.sqrt(0.05), clip=True),
        transforms.Normalize(cifar_mean, cifar_std),
    ]
)


def create_resnet18_model(num_classes):
    """
    Create a ResNet18 model with a custom number of output classes.

    Uses torchvision's ResNet18 without pretraining for fair comparison.

    Args:
        num_classes (int): Number of classes for the output layer.

    Returns:
        torch.nn.Module: The modified ResNet18 model.
    """
    resnet18 = torchvision.models.resnet18(pretrained=False)
    resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, num_classes)
    return resnet18.to(device)


num_classes = 100
resnet18 = create_resnet18_model(num_classes)
summary(resnet18, input_size=(1, 3, 224, 224))


def get_predictions(model, data_loader, device):
    """
    Get predictions from a model on a dataset.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for the dataset.
        device: Device to run inference on.

    Returns:
        tuple: (predictions array, true labels array)

    Raises:
        AssertionError: If outputs shape is unexpected.
    """
    y_pred = []
    y_true = []
    model = model.eval().to(device)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            assert outputs.ndim == 2, f"Expected 2D output, got {outputs.ndim}D"
            assert (
                outputs.shape[0] == images.shape[0]
            ), f"Batch size mismatch: outputs {outputs.shape[0]}, images {images.shape[0]}"
            y_pred.append(outputs.argmax(dim=-1).cpu().numpy())

            y_true.append(labels.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0).flatten()
    y_pred = np.concatenate(y_pred, axis=0).flatten()
    return y_pred, y_true


def evaluate(predictions, actual_values, model_names, class_names=None, plot_cm=False):
    """
    Evaluate multiple models using standard classification metrics.

    Computes accuracy, precision, recall, and F1-score for each model.
    Optionally plots confusion matrices.

    Args:
        predictions: List of prediction arrays.
        actual_values: List of true label arrays.
        model_names: List of model names for indexing.
        class_names: List of class names for confusion matrix labels.
        plot_cm: Whether to plot confusion matrices.

    Returns:
        pd.DataFrame: DataFrame with metrics for each model.

    Raises:
        AssertionError: If lengths don't match.
    """
    assert (
        len(predictions) == len(actual_values) == len(model_names)
    ), "Predictions, actual values, and model names must have the same length."
    scores = defaultdict(lambda: defaultdict(float))
    for i, name in enumerate(model_names):
        scores[name]["Accuracy"] = accuracy_score(actual_values[i], predictions[i])
        scores[name]["Precision"] = precision_score(
            actual_values[i], predictions[i], average="weighted", zero_division=0
        )
        scores[name]["Recall"] = recall_score(
            actual_values[i], predictions[i], average="weighted", zero_division=0
        )
        scores[name]["F1 score"] = f1_score(
            actual_values[i], predictions[i], average="weighted", zero_division=0
        )

        if plot_cm:
            plot_confusion_matrix(
                actual_values[i],
                predictions[i],
                class_names,
                title=f"Confusion Matrix for {name}",
            )

    return pd.DataFrame(scores)


def train_epoch(model, data_loader, criterion, optimizer, device):
    """
    Perform one training epoch.

    Args:
        model: PyTorch model to train.
        data_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for updating weights.
        device: Device to run on.

    Returns:
        tuple: (average loss, accuracy)

    Raises:
        ValueError: If data_loader is empty or model is not on device.
    """
    model.train()
    num_batches = len(data_loader)
    if num_batches == 0:
        raise ValueError("DataLoader is empty.")
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / num_batches, correct / total


def validation_epoch(model, data_loader, criterion, device):
    """
    Perform one validation epoch.

    Args:
        model: PyTorch model to evaluate.
        data_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        tuple: (average loss, accuracy)

    Raises:
        ValueError: If data_loader is empty.
    """
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)
    if num_batches == 0:
        raise ValueError("DataLoader is empty.")
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / num_batches, correct / total


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    device,
    report_val=True,
):
    hist = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    model = model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        hist["train_loss"].append(train_loss)
        hist["train_accuracy"].append(train_accuracy)
        print(
            f"Epoch [{epoch}] Average Training Loss: {train_loss:.4f} Average Training Accuracy: {train_accuracy:.4f}"
        )

        if report_val:
            val_loss, val_accuracy = validation_epoch(
                model, val_loader, criterion, device
            )
            hist["val_loss"].append(val_loss)
            hist["val_accuracy"].append(val_accuracy)
            print(
                f"Epoch [{epoch}] Average Validation Loss: {val_loss:.4f} Average Validation Accuracy: {val_accuracy:.4f}"
            )

    return hist


def plot_history(history, model_name):
    range_epochs = range(1, len(history["val_loss"]) + 1)
    plt.plot(range_epochs, history["train_loss"])
    plt.plot(range_epochs, history["val_loss"])
    plt.title(f"{model_name} Model Loss over Epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.ylim(0, max(max(history["train_loss"]), max(history["val_loss"])) * 1.01)
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()

    plt.plot(range_epochs, history["train_accuracy"])
    plt.plot(range_epochs, history["val_accuracy"])
    plt.title(f"{model_name} Model Accuracy over Epochs")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(["Train", "Val"])
    plt.ylim(0, 1)
    plt.show()


restrainvalset.set_transform(cifar_transform)
restrainset.set_transform(cifar_transform)
resvalset.set_transform(cifar_transform)
restestset.set_transform(cifar_transform)


batch_size = 64
trainvalloader = DataLoader(
    restrainvalset, batch_size=batch_size, shuffle=True, num_workers=2
)
trainloader = DataLoader(
    restrainset, batch_size=batch_size, shuffle=True, num_workers=2
)
valloader = DataLoader(resvalset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(restestset, batch_size=batch_size, shuffle=False, num_workers=2)


epochs = 20
lr = 1e-3

resnet18 = create_resnet18_model(num_classes)

optimizer = torch.optim.Adam(resnet18.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(resnet18, trainloader, valloader, criterion, optimizer, epochs, device)

state_dict = resnet18.state_dict()
torch.save(state_dict, "resnet18.pth")

plot_history(hist, "ResNet18 without noise")


restrainvalset.set_transform(noisy_cifar_transform)
restrainset.set_transform(noisy_cifar_transform)
resvalset.set_transform(noisy_cifar_transform)
restestset.set_transform(noisy_cifar_transform)


epochs = 20
lr = 1e-3

noisy_resnet18 = create_resnet18_model(num_classes)

optimizer = torch.optim.Adam(noisy_resnet18.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(
    noisy_resnet18, trainloader, valloader, criterion, optimizer, epochs, device
)

state_dict = noisy_resnet18.state_dict()
torch.save(state_dict, "noisy_resnet18.pth")

plot_history(hist, "ResNet18 with noise")


restrainvalset.set_transform(cifar_transform)
restrainset.set_transform(cifar_transform)
resvalset.set_transform(cifar_transform)
restestset.set_transform(cifar_transform)


y_pred = []
y_true = []
model_names = ["ResNet18", "Noisy ResNet18"]

pred, true = get_predictions(resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

pred, true = get_predictions(noisy_resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred, y_true, model_names)
scores


flowers102_class_names = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen ",
    "watercress",
    "canna lily",
    "hippeastrum ",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]


vittrainset = torchvision.datasets.Flowers102(
    root="./data", split="train", download=True
)
vitvalset = torchvision.datasets.Flowers102(root="./data", split="val", download=True)
vittestset = torchvision.datasets.Flowers102(root="./data", split="test", download=True)

vittrainset = CustomDataset(vittrainset)
vitvalset = CustomDataset(vitvalset)
vittestset = CustomDataset(vittestset)

len(vittrainset), len(vitvalset), len(vittestset)


n_sample = 5
indices = np.random.choice(len(vittrainset), n_sample)

plt.figure(figsize=(10, 5))
for i in range(n_sample):
    img, label = vittrainset[indices[i]]
    img = np.array(img) / 255.0
    plt.subplot(2, n_sample, i + 1)
    plt.imshow(img)
    plt.title(flowers102_class_names[label])
    plt.axis("off")


mean = (0, 0, 0)
std = (0, 0, 0)
for image, _ in vittrainset:
    img = np.array(image) / 255.0
    mean += img.mean(axis=(0, 1))
    std += img.std(axis=(0, 1))

mean /= len(vittrainset)
std /= len(vittrainset)

flowers_mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
flowers_std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
print(flowers_mean)
print(flowers_std)


flowers_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(flowers_mean, flowers_std),
    ]
)


def create_vit_model(num_classes, pretrained):
    vit = torchvision.models.vit_b_16(weights="DEFAULT" if pretrained else None)

    if pretrained:
        for param in vit.parameters():
            param.requires_grad = False

    vit.heads.head = nn.Linear(vit.heads.head.in_features, num_classes)
    return vit.to(device)


num_classes = 102
vit = create_vit_model(num_classes, True)
summary(vit, input_size=(1, 3, 224, 224))


vittrainset.set_transform(flowers_transform)
vitvalset.set_transform(flowers_transform)
vittestset.set_transform(flowers_transform)


batch_size = 64
trainloader = DataLoader(
    vittrainset, batch_size=batch_size, shuffle=True, num_workers=2
)
valloader = DataLoader(vitvalset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(vittestset, batch_size=batch_size, shuffle=False, num_workers=2)


epochs = 5
lr = 1e-3

pretrained_vit = create_vit_model(num_classes, True)

optimizer = torch.optim.Adam(pretrained_vit.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(
    pretrained_vit, trainloader, valloader, criterion, optimizer, epochs, device
)

state_dict = pretrained_vit.state_dict()
torch.save(state_dict, "pretrained_vit.pth")

plot_history(hist, "Pretrained ViT")


epochs = 10
lr = 1e-3

vit = create_vit_model(num_classes, False)

optimizer = torch.optim.Adam(vit.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(vit, trainloader, valloader, criterion, optimizer, epochs, device)

state_dict = vit.state_dict()
torch.save(state_dict, "vit.pth")

plot_history(hist, "Fully Trained ViT")


y_pred = []
y_true = []
model_names = ["Fully Trained ViT", "Pretrained ViT"]

pred, true = get_predictions(vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

pred, true = get_predictions(pretrained_vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred, y_true, model_names)
scores


class AdversarialDataLoader(DataLoader):
    def __init__(self, *args, attack, attack_args, mean, std, device, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack = attack
        self.attack_args = attack_args
        self.device = device
        self.attack_mode = False
        self.clean_mode = False

        self.mean = mean.to(device)
        self.std = std.to(device)

    def __iter__(self):
        data_iter = super().__iter__()
        for images, labels in data_iter:
            images, labels = images.to(self.device), labels.to(self.device)

            if self.attack_mode and not self.clean_mode:
                adv_images = self.attack(
                    images, labels, self.mean, self.std, self.attack_args
                )
                yield adv_images, labels
            elif self.attack_mode:
                adv_images = self.attack(
                    images, labels, self.mean, self.std, self.attack_args
                )
                yield images, adv_images, labels
            else:
                yield images, labels

    def set_attack_mode(self):
        self.attack_mode = True
        self.clean_mode = False

    def set_clean_mode(self):
        self.attack_mode = False
        self.clean_mode = True

    def set_both_mode(self):
        self.attack_mode = True
        self.clean_mode = True


def plot_adversarial_example(loader, model, class_names, mean, std, device):
    model.eval()
    loader.set_both_mode()
    dataiter = iter(loader)
    plt.figure(figsize=(12, 8))

    while True:
        images, adv_images, labels = next(dataiter)
        image = images[0]
        adv_image = adv_images[0]
        label = labels[0]
        std = std.to(device)
        mean = mean.to(device)

        unscaled_img = image * std + mean
        unscaled_img = unscaled_img.cpu().numpy().clip(0, 1)

        with torch.no_grad():
            logits = model(image.unsqueeze(0))
            probs = logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1).cpu().numpy()

        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        unscaled_adv_image = adv_image * std + mean
        unscaled_adv_image = unscaled_adv_image.cpu().numpy().clip(0, 1)

        noise = np.abs(unscaled_adv_image - unscaled_img)

        with torch.no_grad():
            logits = model(adv_image.unsqueeze(0))
            adv_probs = logits.softmax(dim=-1)
            adv_preds = adv_probs.argmax(dim=-1).cpu().numpy()

        if preds[0] != adv_preds[0]:
            break

    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(unscaled_img, (1, 2, 0)))
    plt.title(
        f"clean image with label: {class_names[label]}\nclassified as: {class_names[preds[0]]}\nconfidence:{probs[0,preds[0]].item():.2f}"
    )
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.transpose(noise, (1, 2, 0)))
    plt.title(f"added noise\nmax value = {noise.max():.2f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(np.transpose(unscaled_adv_image, (1, 2, 0)))
    plt.title(
        f"adversarial example image classified as: {class_names[adv_preds[0]]}\nconfidence:{adv_probs[0,adv_preds[0]].item():.2f}"
    )
    plt.axis("off")


def fgsm_attack(images, labels, mean, std, args):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    model = args["model"]
    epsilon = args["epsilon"]

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    perturbation = epsilon * images.grad.sign()
    images = images * std + mean
    adv_images = images + perturbation
    adv_images = torch.clamp(adv_images, 0, 1).detach()
    adv_images = (adv_images - mean) / std

    return adv_images


attack = fgsm_attack
attack_args = {
    "epsilon": 0.1,
    "alpha": 0.02,
    "steps": 7,
}


restrainvalset.set_transform(cifar_transform)
restrainset.set_transform(cifar_transform)
resvalset.set_transform(cifar_transform)
restestset.set_transform(cifar_transform)


batch_size = 64
trainvalloader = AdversarialDataLoader(
    restrainvalset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    attack=attack,
    mean=cifar_mean,
    std=cifar_std,
    device=device,
    attack_args=attack_args,
)
trainloader = AdversarialDataLoader(
    restrainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    attack=attack,
    mean=cifar_mean,
    std=cifar_std,
    device=device,
    attack_args=attack_args,
)
valloader = AdversarialDataLoader(
    resvalset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=cifar_mean,
    std=cifar_std,
    device=device,
    attack_args=attack_args,
)
testloader = AdversarialDataLoader(
    restestset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=cifar_mean,
    std=cifar_std,
    device=device,
    attack_args=attack_args,
)


resnet18 = create_resnet18_model(100)
noisy_resnet18 = create_resnet18_model(100)

try:
    resnet18.load_state_dict(torch.load("resnet18.pth"))
    print("Successfully loaded ResNet18 model.")
except FileNotFoundError:
    print("Error: 'resnet18.pth' not found. Please train the model first.")
    raise
except Exception as e:
    print(f"Error loading ResNet18 model: {e}")
    raise

try:
    noisy_resnet18.load_state_dict(torch.load("noisy_resnet18.pth"))
    print("Successfully loaded noisy ResNet18 model.")
except FileNotFoundError:
    print("Error: 'noisy_resnet18.pth' not found. Please train the model first.")
    raise
except Exception as e:
    print(f"Error loading noisy ResNet18 model: {e}")
    raise


attack_args["model"] = resnet18
plot_adversarial_example(
    valloader, resnet18, cifar100_class_names, cifar_mean, cifar_std, device
)


attack_args["model"] = noisy_resnet18
plot_adversarial_example(
    valloader, noisy_resnet18, cifar100_class_names, cifar_mean, cifar_std, device
)


trainvalloader.set_attack_mode()
trainloader.set_attack_mode()
valloader.set_attack_mode()
testloader.set_attack_mode()


y_pred = []
y_true = []
model_names = ["ResNet18 FGSM attacked", "Noisy ResNet18 FGSM attacked"]

attack_args["model"] = resnet18
pred, true = get_predictions(resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

attack_args["model"] = noisy_resnet18
pred, true = get_predictions(noisy_resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=cifar100_class_names)
scores


epochs = 10
lr = 1e-3

attack_args["model"] = resnet18
optimizer = torch.optim.Adam(resnet18.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(resnet18, trainloader, valloader, criterion, optimizer, epochs, device)

plot_history(hist, "Adversarially Trained ResNet18 without noise")


epochs = 10
lr = 1e-3

attack_args["model"] = noisy_resnet18
optimizer = torch.optim.Adam(noisy_resnet18.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(
    noisy_resnet18, trainloader, valloader, criterion, optimizer, epochs, device
)

plot_history(hist, "Adversarially Trained ResNet18 with noise")


y_pred = []
y_true = []
model_names = [
    "ResNet18 FGSM attacked",
    "Noisy ResNet18 FGSM attacked",
    "ResNet18 clean",
    "Noisy ResNet18 clean",
]

attack_args["model"] = resnet18
pred, true = get_predictions(resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

attack_args["model"] = noisy_resnet18
pred, true = get_predictions(noisy_resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

trainvalloader.set_clean_mode()
trainloader.set_clean_mode()
valloader.set_clean_mode()
testloader.set_clean_mode()

pred, true = get_predictions(resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

pred, true = get_predictions(noisy_resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=cifar100_class_names)
scores


trainloader = AdversarialDataLoader(
    vittrainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    attack=attack,
    mean=flowers_mean,
    std=flowers_std,
    device=device,
    attack_args=attack_args,
)
valloader = AdversarialDataLoader(
    vitvalset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=flowers_mean,
    std=flowers_std,
    device=device,
    attack_args=attack_args,
)
testloader = AdversarialDataLoader(
    vittestset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=flowers_mean,
    std=flowers_std,
    device=device,
    attack_args=attack_args,
)


vittrainset.set_transform(flowers_transform)
vitvalset.set_transform(flowers_transform)
vittestset.set_transform(flowers_transform)


vit = create_vit_model(102, False)
pretrained_vit = create_vit_model(102, True)

vit.load_state_dict(torch.load("vit.pth"))
pretrained_vit.load_state_dict(torch.load("pretrained_vit.pth"))


attack_args["model"] = vit
plot_adversarial_example(
    valloader, vit, flowers102_class_names, flowers_mean, flowers_std, device
)


attack_args["model"] = pretrained_vit
plot_adversarial_example(
    valloader, pretrained_vit, flowers102_class_names, flowers_mean, flowers_std, device
)


trainloader.set_attack_mode()
valloader.set_attack_mode()
testloader.set_attack_mode()


y_pred = []
y_true = []
model_names = ["Fully Trained ViT FGSM attacked", "Pretrained ViT FGSM attacked"]

attack_args["model"] = vit
pred, true = get_predictions(vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

attack_args["model"] = pretrained_vit
pred, true = get_predictions(pretrained_vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=flowers102_class_names)
scores


epochs = 10
lr = 1e-3

attack_args["model"] = vit
optimizer = torch.optim.Adam(vit.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(vit, trainloader, valloader, criterion, optimizer, epochs, device)

plot_history(hist, "Adversarially Trained ViT")


epochs = 10
lr = 1e-3

attack_args["model"] = pretrained_vit
optimizer = torch.optim.Adam(pretrained_vit.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(
    pretrained_vit, trainloader, valloader, criterion, optimizer, epochs, device
)

plot_history(hist, "Adversarially Trained Pretrained ViT")


y_pred = []
y_true = []
model_names = [
    "ViT FGSM attacked",
    "Pretrained ViT FGSM attacked",
    "ViT clean",
    "Pretrained ViT clean",
]

attack_args["model"] = vit
pred, true = get_predictions(vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

attack_args["model"] = pretrained_vit
pred, true = get_predictions(pretrained_vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

trainvalloader.set_clean_mode()
trainloader.set_clean_mode()
valloader.set_clean_mode()
testloader.set_clean_mode()

pred, true = get_predictions(vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

pred, true = get_predictions(pretrained_vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=flowers102_class_names)
scores


def pgd_attack(images, labels, mean, std, args):
    original_images = images.clone().detach().to(device)
    original_images = original_images * std + mean
    labels = labels.to(device)
    adv_images = original_images.clone().detach()

    model = args["model"].eval()
    alpha = args["alpha"]
    epsilon = args["epsilon"]
    steps = args["steps"]
    criterion = nn.CrossEntropyLoss()

    for i in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            delta = torch.clamp(adv_images - original_images, -epsilon, epsilon)
            adv_images = torch.clamp(original_images + delta, 0, 1)

        adv_images = adv_images.detach()

    adv_images = (adv_images - mean) / std
    return adv_images


attack = pgd_attack


trainvalloader = AdversarialDataLoader(
    restrainvalset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    attack=attack,
    mean=cifar_mean,
    std=cifar_std,
    device=device,
    attack_args=attack_args,
)
trainloader = AdversarialDataLoader(
    restrainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    attack=attack,
    mean=cifar_mean,
    std=cifar_std,
    device=device,
    attack_args=attack_args,
)
valloader = AdversarialDataLoader(
    resvalset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=cifar_mean,
    std=cifar_std,
    device=device,
    attack_args=attack_args,
)
testloader = AdversarialDataLoader(
    restestset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=cifar_mean,
    std=cifar_std,
    device=device,
    attack_args=attack_args,
)


restrainvalset.set_transform(cifar_transform)
restrainset.set_transform(cifar_transform)
resvalset.set_transform(cifar_transform)
restestset.set_transform(cifar_transform)

resnet18 = create_resnet18_model(100)
noisy_resnet18 = create_resnet18_model(100)

resnet18.load_state_dict(torch.load("resnet18.pth"))
noisy_resnet18.load_state_dict(torch.load("noisy_resnet18.pth"))


attack_args["model"] = resnet18
plot_adversarial_example(
    valloader, resnet18, cifar100_class_names, cifar_mean, cifar_std, device
)


attack_args["model"] = noisy_resnet18
plot_adversarial_example(
    valloader, noisy_resnet18, cifar100_class_names, cifar_mean, cifar_std, device
)


trainvalloader.set_attack_mode()
trainloader.set_attack_mode()
valloader.set_attack_mode()
testloader.set_attack_mode()


y_pred = []
y_true = []
model_names = ["ResNet18 PGD attacked", "Noisy ResNet18 PGD attacked"]

attack_args["model"] = resnet18
pred, true = get_predictions(resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

attack_args["model"] = noisy_resnet18
pred, true = get_predictions(noisy_resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=cifar100_class_names)
scores


epochs = 10
lr = 1e-3

attack_args["model"] = resnet18
optimizer = torch.optim.Adam(resnet18.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(resnet18, trainloader, valloader, criterion, optimizer, epochs, device)

plot_history(hist, "Adversarially Trained ResNet18 without noise")


epochs = 10
lr = 1e-3

attack_args["model"] = noisy_resnet18
optimizer = torch.optim.Adam(noisy_resnet18.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(
    noisy_resnet18, trainloader, valloader, criterion, optimizer, epochs, device
)

plot_history(hist, "Adversarially Trained ResNet18 with noise")


y_pred = []
y_true = []
model_names = [
    "ResNet18 PGD attacked",
    "Noisy ResNet18 PGD attacked",
    "ResNet18 clean",
    "Noisy ResNet18 clean",
]

attack_args["model"] = resnet18
pred, true = get_predictions(resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

attack_args["model"] = noisy_resnet18
pred, true = get_predictions(noisy_resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

trainvalloader.set_clean_mode()
trainloader.set_clean_mode()
valloader.set_clean_mode()
testloader.set_clean_mode()

pred, true = get_predictions(resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

pred, true = get_predictions(noisy_resnet18, valloader, device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=cifar100_class_names)
scores


trainloader = AdversarialDataLoader(
    vittrainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    attack=attack,
    mean=flowers_mean,
    std=flowers_std,
    device=device,
    attack_args=attack_args,
)
valloader = AdversarialDataLoader(
    vitvalset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=flowers_mean,
    std=flowers_std,
    device=device,
    attack_args=attack_args,
)
testloader = AdversarialDataLoader(
    vittestset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=flowers_mean,
    std=flowers_std,
    device=device,
    attack_args=attack_args,
)


vittrainset.set_transform(flowers_transform)
vitvalset.set_transform(flowers_transform)
vittestset.set_transform(flowers_transform)

vit = create_vit_model(102, False)
pretrained_vit = create_vit_model(102, True)

vit.load_state_dict(torch.load("vit.pth"))
pretrained_vit.load_state_dict(torch.load("pretrained_vit.pth"))


attack_args["model"] = vit
plot_adversarial_example(
    valloader, vit, flowers102_class_names, flowers_mean, flowers_std, device
)


attack_args["model"] = pretrained_vit
plot_adversarial_example(
    valloader, pretrained_vit, flowers102_class_names, flowers_mean, flowers_std, device
)


trainvalloader.set_attack_mode()
trainloader.set_attack_mode()
valloader.set_attack_mode()
testloader.set_attack_mode()


y_pred = []
y_true = []
model_names = ["Fully Trained ViT PGD attacked", "Pretrained ViT PGD attacked"]

attack_args["model"] = vit
pred, true = get_predictions(vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

attack_args["model"] = pretrained_vit
pred, true = get_predictions(pretrained_vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=flowers102_class_names)
scores


epochs = 10
lr = 1e-3

attack_args["model"] = vit
optimizer = torch.optim.Adam(vit.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(vit, trainloader, valloader, criterion, optimizer, epochs, device)

plot_history(hist, "Adversarially Trained ViT")


epochs = 10
lr = 1e-3

attack_args["model"] = pretrained_vit
optimizer = torch.optim.Adam(pretrained_vit.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(
    pretrained_vit, trainloader, valloader, criterion, optimizer, epochs, device
)

plot_history(hist, "Adversarially Trained Pretrained ViT")


y_pred = []
y_true = []
model_names = [
    "ViT PGD attacked",
    "Pretrained ViT PGD attacked",
    "ViT clean",
    "Pretrained ViT clean",
]

attack_args["model"] = vit
pred, true = get_predictions(vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

attack_args["model"] = pretrained_vit
pred, true = get_predictions(pretrained_vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

trainvalloader.set_clean_mode()
trainloader.set_clean_mode()
valloader.set_clean_mode()
testloader.set_clean_mode()

pred, true = get_predictions(vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

pred, true = get_predictions(pretrained_vit, valloader, device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=flowers102_class_names)
scores


attack = fgsm_attack
attack_args = {
    "epsilon": 0.1,
    "alpha": 0.02,
    "steps": 7,
}

restrainset.set_transform(cifar_transform)
restestset.set_transform(cifar_transform)

batch_size = 64
trainloader = AdversarialDataLoader(
    restrainset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=cifar_mean,
    std=cifar_std,
    device=device,
    attack_args=attack_args,
)
testloader = AdversarialDataLoader(
    restestset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=cifar_mean,
    std=cifar_std,
    device=device,
    attack_args=attack_args,
)
testloader_pgd = AdversarialDataLoader(
    restestset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=pgd_attack,
    mean=cifar_mean,
    std=cifar_std,
    device=device,
    attack_args=attack_args,
)

trainloader.set_attack_mode()
testloader.set_attack_mode()
testloader_pgd.set_attack_mode()


epochs = 10
lr = 1e-3

resnet18 = create_resnet18_model(100)
resnet18.load_state_dict(torch.load("resnet18.pth"))

attack_args["model"] = resnet18
optimizer = torch.optim.Adam(resnet18.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train(resnet18, trainloader, testloader, criterion, optimizer, epochs, device, False)

state_dict = resnet18.state_dict()
torch.save(state_dict, "test_resnet18.pth")


untrained_resnet18 = create_resnet18_model(100)
untrained_resnet18.load_state_dict(torch.load("resnet18.pth"))


y_pred = []
y_true = []
models = [untrained_resnet18, resnet18]
model_names = [
    "ResNet18 clean",
    "ResNet18 FGSM",
    "ResNet18 PGD",
    "Adv. Trained ResNet18 clean",
    "Adv. Trained ResNet18 FGSM",
    "Adv. Trained ResNet18 PGD",
]

for model in models:
    testloader.set_clean_mode()
    attack_args["model"] = model
    pred, true = get_predictions(model, testloader, device)
    y_pred.append(pred)
    y_true.append(true)

    testloader.set_attack_mode()
    attack_args["model"] = model
    pred, true = get_predictions(model, testloader, device)
    y_pred.append(pred)
    y_true.append(true)

    attack_args["model"] = model
    pred, true = get_predictions(model, testloader_pgd, device)
    y_pred.append(pred)
    y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=cifar100_class_names)
scores


epochs = 10
lr = 1e-3
testloader.set_attack_mode()

noisy_resnet18 = create_resnet18_model(100)
noisy_resnet18.load_state_dict(torch.load("noisy_resnet18.pth"))

attack_args["model"] = noisy_resnet18
optimizer = torch.optim.Adam(noisy_resnet18.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train(
    noisy_resnet18, trainloader, testloader, criterion, optimizer, epochs, device, False
)

state_dict = noisy_resnet18.state_dict()
torch.save(state_dict, "test_noisy_resnet18.pth")


untrained_resnet18 = create_resnet18_model(100)
untrained_resnet18.load_state_dict(torch.load("noisy_resnet18.pth"))


y_pred = []
y_true = []
models = [untrained_resnet18, noisy_resnet18]
model_names = [
    "ResNet18 clean",
    "ResNet18 FGSM",
    "ResNet18 PGD",
    "Adv. Trained ResNet18 clean",
    "Adv. Trained ResNet18 FGSM",
    "Adv. Trained ResNet18 PGD",
]

for model in models:
    testloader.set_clean_mode()
    attack_args["model"] = model
    pred, true = get_predictions(model, testloader, device)
    y_pred.append(pred)
    y_true.append(true)

    testloader.set_attack_mode()
    attack_args["model"] = model
    pred, true = get_predictions(model, testloader, device)
    y_pred.append(pred)
    y_true.append(true)

    attack_args["model"] = model
    pred, true = get_predictions(model, testloader_pgd, device)
    y_pred.append(pred)
    y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=cifar100_class_names)
scores


attack = fgsm_attack
attack_args = {
    "epsilon": 0.1,
    "alpha": 0.02,
    "steps": 7,
}

vittrainset.set_transform(flowers_transform)
vittestset.set_transform(flowers_transform)

batch_size = 64
trainloader = AdversarialDataLoader(
    vittrainset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=flowers_mean,
    std=flowers_std,
    device=device,
    attack_args=attack_args,
)
testloader = AdversarialDataLoader(
    vittestset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    mean=flowers_mean,
    std=flowers_std,
    device=device,
    attack_args=attack_args,
)
testloader_pgd = AdversarialDataLoader(
    vittestset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=pgd_attack,
    mean=flowers_mean,
    std=flowers_std,
    device=device,
    attack_args=attack_args,
)

trainloader.set_attack_mode()
testloader.set_attack_mode()
testloader_pgd.set_attack_mode()


epochs = 10
lr = 1e-3

pretrained_vit = create_vit_model(102, True)
pretrained_vit.load_state_dict(torch.load("pretrained_vit.pth"))

attack_args["model"] = pretrained_vit
optimizer = torch.optim.Adam(pretrained_vit.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

hist = train(
    pretrained_vit, trainloader, testloader, criterion, optimizer, epochs, device
)
plot_history(hist, "Pretrained ViT test images")

state_dict = pretrained_vit.state_dict()
torch.save(state_dict, "test_pretrained_vit.pth")


untrained_vit = create_vit_model(102, True)
untrained_vit.load_state_dict(torch.load("pretrained_vit.pth"))


y_pred = []
y_true = []
models = [untrained_vit, pretrained_vit]
model_names = [
    "Pre. ViT clean",
    "Pre. ViT FGSM",
    "Pre. ViT PGD",
    "Adv. Trained Pre. ViT clean",
    "Adv. Trained Pre. ViT FGSM",
    "Adv. Trained Pre. ViT PGD",
]

for model in models:
    testloader.set_clean_mode()
    attack_args["model"] = model
    pred, true = get_predictions(model, testloader, device)
    y_pred.append(pred)
    y_true.append(true)

    testloader.set_attack_mode()
    attack_args["model"] = model
    pred, true = get_predictions(model, testloader, device)
    y_pred.append(pred)
    y_true.append(true)

    attack_args["model"] = model
    pred, true = get_predictions(model, testloader_pgd, device)
    y_pred.append(pred)
    y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=flowers102_class_names)
scores


epochs = 10
lr = 1e-3

vit = create_vit_model(102, False)
vit.load_state_dict(torch.load("vit.pth"))

attack_args["model"] = vit
optimizer = torch.optim.Adam(vit.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train(vit, trainloader, testloader, criterion, optimizer, epochs, device, False)

state_dict = vit.state_dict()
torch.save(state_dict, "test_vit.pth")


untrained_vit = create_vit_model(102, False)
untrained_vit.load_state_dict(torch.load("vit.pth"))


y_pred = []
y_true = []
models = [untrained_vit, vit]
model_names = [
    "ViT clean",
    "ViT FGSM",
    "ViT PGD",
    "Adv. Trained ViT clean",
    "Adv. Trained ViT FGSM",
    "Adv. Trained ViT PGD",
]

for model in models:
    testloader.set_clean_mode()
    attack_args["model"] = model
    pred, true = get_predictions(model, testloader, device)
    y_pred.append(pred)
    y_true.append(true)

    testloader.set_attack_mode()
    attack_args["model"] = model
    pred, true = get_predictions(model, testloader, device)
    y_pred.append(pred)
    y_true.append(true)

    attack_args["model"] = model
    pred, true = get_predictions(model, testloader_pgd, device)
    y_pred.append(pred)
    y_true.append(true)

scores = evaluate(y_pred, y_true, model_names, class_names=flowers102_class_names)
scores


def get_grad_cam(model, layer, image, mean, std, label, reshape_transform):
    model.eval()
    cam = GradCAM(
        model=model, target_layers=[layer], reshape_transform=reshape_transform
    )
    image = image.unsqueeze(0).to(device)

    output = model(image)
    pred_class = output.argmax().item()

    grayscale_cam = cam(
        input_tensor=image, targets=[ClassifierOutputTarget(label.item())]
    )[0]

    image = (image * std) + mean
    img_np = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_np = img_np.clip(0, 1)

    return show_cam_on_image(img_np, grayscale_cam, use_rgb=True), pred_class


def plot_image(
    plt, idx, image, model_name, image_name, class_names, pred, row=5, height=3
):
    plt.subplot(row, height, idx)
    plt.imshow(image)
    plt.title(model_name + "\nClassified as:" + class_names[pred] + "\n" + image_name)
    plt.axis("off")


def plot_grad_cams(
    model,
    start_idx,
    layer,
    mean,
    std,
    model_name,
    class_names,
    attack_args,
    testloader,
    testloader_pgd,
    data_idx=0,
    reshape_transform=None,
):
    attack_args["model"] = model

    testloader.set_both_mode()
    dataiter = iter(testloader)
    testloader_pgd.set_attack_mode()
    dataiter_pgd = iter(testloader_pgd)

    images, fgsms, labels = next(dataiter)
    pgds, labels = next(dataiter_pgd)

    image = images[data_idx]
    fgsm = fgsms[data_idx]
    pgd = pgds[data_idx]
    label = labels[data_idx]

    res_image, pred = get_grad_cam(
        model, layer, image, mean, std, label, reshape_transform
    )
    plot_image(plt, start_idx, res_image, model_name, "Clean", class_names, pred)

    res_fgsm, pred = get_grad_cam(
        model, layer, fgsm, mean, std, label, reshape_transform
    )
    plot_image(plt, start_idx + 1, res_fgsm, model_name, "FGSM", class_names, pred)

    res_pgd, pred = get_grad_cam(model, layer, pgd, mean, std, label, reshape_transform)
    plot_image(plt, start_idx + 2, res_pgd, model_name, "PGD", class_names, pred)


resnet18 = create_resnet18_model(100)
resnet18.load_state_dict(torch.load("test_resnet18.pth"))
noisy_resnet18 = create_resnet18_model(100)
noisy_resnet18.load_state_dict(torch.load("test_noisy_resnet18.pth"))


testloader.set_both_mode()
dataiter = iter(testloader)
testloader_pgd.set_attack_mode()
dataiter_pgd = iter(testloader_pgd)

attack_args["model"] = resnet18
images, fgsms, labels = next(dataiter)
pgds, labels = next(dataiter_pgd)

mean = cifar_mean.to(device)
std = cifar_std.to(device)

names = cifar100_class_names
plt.figure(figsize=(9, 18))

idx = 1
image = images[idx]
fgsm = fgsms[idx]
pgd = pgds[idx]
label = labels[idx]

names = cifar100_class_names
plt.figure(figsize=(9, 18))

unscaled_img = (image * std + mean).clip(0, 1)
unscaled_fgsm = (fgsm * std + mean).clip(0, 1)
unscaled_pgd = (pgd * std + mean).clip(0, 1)

unscaled_img = unscaled_img.cpu().numpy()
unscaled_fgsm = unscaled_fgsm.cpu().numpy()
unscaled_pgd = unscaled_pgd.cpu().numpy()

plot_image(
    plt, 1, np.transpose(unscaled_img, (1, 2, 0)), "Image", "Clean", names, label
)
plot_image(
    plt, 2, np.transpose(unscaled_fgsm, (1, 2, 0)), "Image", "FGSM", names, label
)
plot_image(plt, 3, np.transpose(unscaled_pgd, (1, 2, 0)), "Image", "PGD", names, label)

untrained_resnet18 = create_resnet18_model(100)
untrained_resnet18.load_state_dict(torch.load("resnet18.pth"))
layer = untrained_resnet18.layer4[-1]
plot_grad_cams(
    untrained_resnet18,
    4,
    layer,
    mean,
    std,
    "ResNet18",
    names,
    attack_args,
    testloader,
    testloader_pgd,
    idx,
)

untrained_resnet18 = create_resnet18_model(100)
untrained_resnet18.load_state_dict(torch.load("noisy_resnet18.pth"))
layer = untrained_resnet18.layer4[-1]
plot_grad_cams(
    untrained_resnet18,
    7,
    layer,
    mean,
    std,
    "Noisy ResNet18",
    names,
    attack_args,
    testloader,
    testloader_pgd,
    idx,
)

layer = resnet18.layer4[-1]
plot_grad_cams(
    resnet18,
    10,
    layer,
    mean,
    std,
    "Adv. ResNet18",
    names,
    attack_args,
    testloader,
    testloader_pgd,
    idx,
)

layer = noisy_resnet18.layer4[-1]
plot_grad_cams(
    noisy_resnet18,
    13,
    layer,
    mean,
    std,
    "Adv. Noisy ResNet18",
    names,
    attack_args,
    testloader,
    testloader_pgd,
    idx,
)


vit = create_vit_model(102, False)
vit.load_state_dict(torch.load("test_vit.pth"))


pretrained_vit = create_vit_model(102, True)
pretrained_vit.load_state_dict(torch.load("test_pretrained_vit.pth"))


def vit_reshape_transform(tensor):
    if tensor.ndim == 3 and tensor.shape[1] > 1:
        tensor = tensor[:, 1:, :]

    batch_size, num_patches, embedding_dim = tensor.shape

    grid_size = int(num_patches**0.5)
    tensor = tensor.reshape(batch_size, grid_size, grid_size, embedding_dim)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


testloader.set_both_mode()
dataiter = iter(testloader)
testloader_pgd.set_attack_mode()
dataiter_pgd = iter(testloader_pgd)

mean = flowers_mean.to(device)
std = flowers_std.to(device)

names = flowers102_class_names
plt.figure(figsize=(9, 18))

attack_args["model"] = pretrained_vit
images, fgsms, labels = next(dataiter)
pgds, labels = next(dataiter_pgd)

idx = 7
image = images[idx]
fgsm = fgsms[idx]
pgd = pgds[idx]
label = labels[idx]

unscaled_img = (image * std + mean).clip(0, 1)
unscaled_fgsm = (fgsm * std + mean).clip(0, 1)
unscaled_pgd = (pgd * std + mean).clip(0, 1)

unscaled_img = unscaled_img.cpu().numpy()
unscaled_fgsm = unscaled_fgsm.cpu().numpy()
unscaled_pgd = unscaled_pgd.cpu().numpy()

plot_image(
    plt, 1, np.transpose(unscaled_img, (1, 2, 0)), "Image", "Clean", names, label
)
plot_image(
    plt, 2, np.transpose(unscaled_fgsm, (1, 2, 0)), "Image", "FGSM", names, label
)
plot_image(plt, 3, np.transpose(unscaled_pgd, (1, 2, 0)), "Image", "PGD", names, label)

untrained_vit = create_vit_model(102, True)
untrained_vit.load_state_dict(torch.load("pretrained_vit.pth"))
for param in untrained_vit.parameters():
    param.requires_grad = True
layer = untrained_vit.encoder.layers[-1].ln_1
plot_grad_cams(
    untrained_vit,
    4,
    layer,
    mean,
    std,
    "Pretrained ViT",
    names,
    attack_args,
    testloader,
    testloader_pgd,
    idx,
    vit_reshape_transform,
)

untrained_vit = create_vit_model(102, False)
untrained_vit.load_state_dict(torch.load("vit.pth"))
layer = untrained_vit.encoder.layers[-1].ln_1
plot_grad_cams(
    untrained_vit,
    7,
    layer,
    mean,
    std,
    "ViT",
    names,
    attack_args,
    testloader,
    testloader_pgd,
    idx,
    vit_reshape_transform,
)

for param in pretrained_vit.parameters():
    param.requires_grad = True
layer = pretrained_vit.encoder.layers[-1].ln_1
plot_grad_cams(
    pretrained_vit,
    10,
    layer,
    mean,
    std,
    "Adv. Pretrained ViT",
    names,
    attack_args,
    testloader,
    testloader_pgd,
    idx,
    vit_reshape_transform,
)

layer = vit.encoder.layers[-1].ln_1
plot_grad_cams(
    vit,
    13,
    layer,
    mean,
    std,
    "Adv. ViT",
    names,
    attack_args,
    testloader,
    testloader_pgd,
    idx,
    vit_reshape_transform,
)
