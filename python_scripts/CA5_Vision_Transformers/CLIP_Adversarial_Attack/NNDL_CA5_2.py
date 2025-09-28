import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def evaluate_clip_attack(
    model, processor, images, texts, original_labels, adv_images=None, adv_texts=None
):
    """Evaluate CLIP performance under adversarial attacks."""
    device = next(model.parameters()).device

    with torch.no_grad():
        image_inputs = processor(
            images=images, return_tensors="pt", do_rescale=False
        ).to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(
            device
        )
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(image_features, text_features.t())
        original_preds = similarity.argmax(dim=1)

    results = {
        "original_accuracy": (original_preds == original_labels).float().mean().item(),
        "original_preds": original_preds,
    }

    if adv_images is not None:
        adv_image_inputs = processor(
            images=adv_images, return_tensors="pt", do_rescale=False
        ).to(device)
        adv_image_features = model.get_image_features(**adv_image_inputs)
        adv_image_features = adv_image_features / adv_image_features.norm(
            dim=-1, keepdim=True
        )

        adv_similarity = torch.matmul(adv_image_features, text_features.t())
        adv_preds = adv_similarity.argmax(dim=1)

        results["adversarial_accuracy"] = (
            (adv_preds == original_labels).float().mean().item()
        )
        results["attack_success_rate"] = 1 - results["adversarial_accuracy"]
        results["adv_preds"] = adv_preds

        cos_sim_original = F.cosine_similarity(
            image_features, text_features[original_labels], dim=1
        )
        cos_sim_adv = F.cosine_similarity(
            adv_image_features, text_features[original_labels], dim=1
        )
        results["avg_similarity_drop"] = (cos_sim_original - cos_sim_adv).mean().item()

    print(f"Original Accuracy: {results['original_accuracy']:.4f}")
    if "attack_success_rate" in results:
        print(f"Attack Success Rate: {results['attack_success_rate']:.4f}")
        print(f"Average Similarity Drop: {results['avg_similarity_drop']:.4f}")

    return results


def visualize_clip_adversarial(
    original_images, adv_images, texts, original_labels, adv_labels, class_names
):
    """Visualize CLIP adversarial examples."""
    n_samples = min(5, len(original_images))
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))

    for i in range(n_samples):
        img_orig = original_images[i].permute(1, 2, 0).cpu().numpy()
        img_orig = np.clip(img_orig, 0, 1)
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title(
            f"Original\n{original_labels[i]}: {class_names[original_labels[i]]}"
        )
        axes[0, i].axis("off")

        img_adv = adv_images[i].permute(1, 2, 0).cpu().numpy()
        img_adv = np.clip(img_adv, 0, 1)
        axes[1, i].imshow(img_adv)
        axes[1, i].set_title(
            f"Adversarial\n{adv_labels[i]}: {class_names[adv_labels[i]]}"
        )
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import Subset, random_split, DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.metrics import precision_recall_fscore_support as score
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from collections import defaultdict
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import torchattacks
import seaborn as sns
import pandas as pd

np.random.seed(42)


classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

cifar_mean = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(3, 1, 1)
cifar_std = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(3, 1, 1)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(clip_mean, clip_std),
    ]
)


num_train = 10000
train_size = int(0.8 * num_train)
val_size = num_train - train_size
batch_size = 64

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
indices = np.random.choice(len(trainset), num_train)
trainvalset = Subset(trainset, indices)
train_subset, val_subset = random_split(trainvalset, [train_size, val_size])
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

len(train_subset), len(testset)


n_sample = 5

plt.figure(figsize=(10, 8))
for i in range(n_sample):
    img, label = trainset[i]
    img = img * clip_std + clip_mean
    plt.subplot(1, n_sample, i + 1)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.title(classes[label])
    plt.axis("off")


target_model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
)
target_model = target_model.to(device)
target_model.eval()
print()


model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()
processor = CLIPProcessor.from_pretrained(model_name)


class_texts = [f"a photo of a {c}" for c in classes]
inputs = processor(text=class_texts, return_tensors="pt", padding=True).to(device)
text_features = model.get_text_features(**inputs).to(device)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
text_features = text_features.detach()


class AdversarialDataLoader(DataLoader):
    def __init__(self, *args, attack, clean_args, adv_args, device, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack = attack
        self.device = device
        self.attack_mode = False
        self.clean_mode = False

        self.clean_mean = clean_args["mean"].to(device)
        self.clean_std = clean_args["std"].to(device)
        self.clean_size = clean_args["size"]

        self.adv_mean = adv_args["mean"].to(device)
        self.adv_std = adv_args["std"].to(device)
        self.adv_size = adv_args["size"]

    def _apply_attack(self, images, labels):
        images = images * self.clean_std + self.clean_mean
        images = F.interpolate(
            images, size=self.adv_size, mode="bilinear", align_corners=False
        )
        images = (images - self.adv_mean) / self.adv_std

        images = self.attack(images, labels)

        images = images * self.adv_std + self.adv_mean
        images = F.interpolate(
            images, size=self.clean_size, mode="bilinear", align_corners=False
        )
        images = (images - self.clean_mean) / self.clean_std

        return images

    def __iter__(self):
        data_iter = super().__iter__()
        for images, labels in data_iter:
            images, labels = images.to(self.device), labels.to(self.device)

            if self.attack_mode and not self.clean_mode:
                images = self._apply_attack(images, labels)
                yield images, labels
            elif self.attack_mode:
                adv_images = self._apply_attack(images, labels)
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


attack = torchattacks.PGD(
    target_model, eps=8 / 255, alpha=2 / 255, steps=7, random_start=True
)
attack.set_normalization_used(mean=cifar_mean, std=cifar_std)
clean_args = {"mean": clip_mean, "std": clip_std, "size": 224}
adv_args = {"mean": cifar_mean, "std": cifar_std, "size": 32}


trainvalloader = AdversarialDataLoader(
    trainvalset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    attack=attack,
    clean_args=clean_args,
    adv_args=adv_args,
    device=device,
)
trainloader = AdversarialDataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    attack=attack,
    clean_args=clean_args,
    adv_args=adv_args,
    device=device,
)
valloader = AdversarialDataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    clean_args=clean_args,
    adv_args=adv_args,
    device=device,
)
testloader = AdversarialDataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    clean_args=clean_args,
    adv_args=adv_args,
    device=device,
)


def get_predictions(model, data_loader, text_features, device):
    y_pred = []
    y_true = []
    model = model.eval().to(device)

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = torch.matmul(image_features, text_features.T)
            probs = logits.softmax(dim=-1)
            y_pred.append(logits.argmax(dim=-1).cpu().numpy())

        y_true.append(labels.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0).flatten()
    y_pred = np.concatenate(y_pred, axis=0).flatten()
    return y_pred, y_true


def evaluate(predictions, actual_values, classes):
    scores = defaultdict(lambda: defaultdict(float))
    precision, recall, fscore, support = score(actual_values, predictions)

    def update_scores(metric, name):
        for i, c in enumerate(classes):
            scores[name][c] = metric[i]

    metric_name = "Accuracy"
    scores[metric_name]["Micro"] = accuracy_score(actual_values, predictions)
    scores[metric_name]["Macro"] = accuracy_score(actual_values, predictions)
    scores[metric_name]["Wieghted"] = accuracy_score(actual_values, predictions)

    cm = confusion_matrix(actual_values, predictions)
    per_class_accuracies = []
    for idx, cls in enumerate(classes):
        true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        true_positives = cm[idx, idx]
        per_class_accuracies.append((true_positives + true_negatives) / np.sum(cm))
    update_scores(per_class_accuracies, metric_name)

    metric_name = "Precision"
    update_scores(precision, metric_name)
    scores[metric_name]["Micro"] = precision_score(
        actual_values, predictions, average="micro"
    )
    scores[metric_name]["Macro"] = precision_score(
        actual_values, predictions, average="macro"
    )
    scores[metric_name]["Wieghted"] = precision_score(
        actual_values, predictions, average="weighted"
    )

    metric_name = "Recall"
    update_scores(recall, metric_name)
    scores[metric_name]["Micro"] = recall_score(
        actual_values, predictions, average="micro"
    )
    scores[metric_name]["Macro"] = recall_score(
        actual_values, predictions, average="macro"
    )
    scores[metric_name]["Wieghted"] = recall_score(
        actual_values, predictions, average="weighted"
    )

    metric_name = "F1 score"
    update_scores(fscore, metric_name)
    scores[metric_name]["Micro"] = f1_score(actual_values, predictions, average="micro")
    scores[metric_name]["Macro"] = f1_score(actual_values, predictions, average="macro")
    scores[metric_name]["Wieghted"] = f1_score(
        actual_values, predictions, average="weighted"
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="RdYlGn", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()

    return pd.DataFrame(scores)


y_pred, y_true = get_predictions(model, valloader, text_features, device)
scores = evaluate(y_pred, y_true, classes)


scores.round(4)


valloader.set_attack_mode()
y_pred, y_true = get_predictions(model, valloader, text_features, device)
scores = evaluate(y_pred, y_true, classes)
valloader.set_clean_mode()


scores.round(4)


valloader.set_both_mode()
dataiter = iter(valloader)


plt.figure(figsize=(12, 8))
images, adv_images, labels = next(dataiter)
image = images[0]
adv_image = adv_images[0]
label = labels[0]

plt.subplot(1, 3, 1)
unscaled_img = image * clip_std.to(device) + clip_mean.to(device)
plt.imshow(np.transpose(unscaled_img.cpu().numpy(), (1, 2, 0)))

with torch.no_grad():
    image_features = model.get_image_features(pixel_values=image.unsqueeze(0))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logits = torch.matmul(image_features, text_features.T)
    probs = logits.softmax(dim=-1)
    preds = probs.argmax(dim=-1).cpu().numpy()

    plt.title(
        f"clean image classified as: {classes[preds[0]]}\nconfidence:{probs[0,preds[0]].item():.2f}"
    )
    plt.axis("off")

plt.subplot(1, 3, 2)
image = image.unsqueeze(0).to(device)
label = label.unsqueeze(0).to(device)

unscaled_adv_image = adv_image.cpu() * clip_std + clip_mean
unscaled_adv_image = unscaled_adv_image.numpy()

noise = np.abs(unscaled_adv_image - unscaled_img.cpu().numpy())
plt.imshow(10 * np.transpose(noise, (1, 2, 0)))
plt.title(f"10x added noise\nmax value = {noise.max():.2f}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(np.transpose(unscaled_adv_image, (1, 2, 0)))

with torch.no_grad():
    image_features = model.get_image_features(pixel_values=adv_image.unsqueeze(0))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logits = torch.matmul(image_features, text_features.T)
    probs = logits.softmax(dim=-1)
    preds = probs.argmax(dim=-1).cpu().numpy()

    plt.title(
        f"adversarial example image classified as: {classes[preds[0]]}\nconfidence:{probs[0,preds[0]].item():.2f}"
    )
    plt.axis("off")


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type=TaskType.FEATURE_EXTRACTION,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        image_features = model.get_image_features(pixel_values=images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        loss = criterion(image_features, labels, text_features)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


def validation_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            loss = criterion(image_features, labels, text_features)

            total_loss += loss.item()

    return total_loss / num_batches


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
        "val_loss": [],
    }

    model = model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        hist["train_loss"].append(train_loss)
        print(f"Epoch [{epoch}] Average Training Loss: {train_loss:.4f}")

        if report_val:
            val_loss = validation_epoch(model, val_loader, criterion, device)
            hist["val_loss"].append(val_loss)
            print(f"Epoch [{epoch}] Average Validation Loss: {val_loss:.4f}")

    return hist


wrapped_criterion = nn.CrossEntropyLoss()


def criterion_wrapper(image_features, labels, text_features):
    return wrapped_criterion(torch.matmul(image_features, text_features.T), labels)


def plot_history(history, model_name):
    plt.plot(history["train_loss"])
    plt.plot(history["val_loss"])
    plt.title(f"{model_name} Model Loss over Epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.ylim(0, max(max(history["train_loss"]), max(history["val_loss"])))
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


trainloader.set_attack_mode()
trainvalloader.set_attack_mode()
valloader.set_attack_mode()
testloader.set_attack_mode()


epochs = 10
momentum = 0.9
lr = 1e-3
weight_decay = 0.0001

optimizer = torch.optim.SGD(
    model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
hist = train(
    model, trainloader, valloader, criterion_wrapper, optimizer, epochs, device
)
plot_history(hist, "Standard Adversarial Trained")


y_pred, y_true = get_predictions(model, valloader, text_features, device)
scores = evaluate(y_pred, y_true, classes)


scores.round(4)


temperature = 0.01


def tecoa_loss(image_features, labels, text_features):
    text_features = text_features[labels]
    similarities = logits = torch.matmul(image_features, text_features.T) / temperature
    targets = torch.arange(image_features.size(0), device=image_features.device)
    loss = F.cross_entropy(similarities, targets)
    return loss


model2 = CLIPModel.from_pretrained(model_name).to(device)
model2 = get_peft_model(model2, lora_config)

optimizer = torch.optim.SGD(
    model2.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
criterion = tecoa_loss
hist2 = train(model2, trainloader, valloader, criterion, optimizer, epochs, device)
plot_history(hist2, "TeCoA Loss wit htemperature = 0.01 Trained")


y_pred, y_true = get_predictions(model2, valloader, text_features, device)
scores = evaluate(y_pred, y_true, classes)


scores.round(4)


temperature = 0.1
model3 = CLIPModel.from_pretrained(model_name).to(device)
model3 = get_peft_model(model3, lora_config)

optimizer = torch.optim.SGD(
    model3.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
criterion = tecoa_loss
hist3 = train(model3, trainloader, valloader, criterion, optimizer, epochs, device)
plot_history(hist3, "TeCoA Loss with temperature = 0.1 Trained")


y_pred, y_true = get_predictions(model3, valloader, text_features, device)
scores = evaluate(y_pred, y_true, classes)


scores.round(4)


class VPT_CLIP(nn.Module):
    def __init__(self, image_size, model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        self.image_size = image_size
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.learned_bias = nn.Parameter(torch.zeros(3, image_size, image_size))

    def get_image_features(self, pixel_values):
        x = pixel_values + self.learned_bias.unsqueeze(0)
        return self.clip_model.get_image_features(pixel_values=x)


temperature = 0.01
model4 = VPT_CLIP(224).to(device)

optimizer = torch.optim.SGD(
    model4.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
criterion = tecoa_loss
hist4 = train(model4, trainloader, valloader, criterion, optimizer, epochs, device)
plot_history(hist4, "TeCoA Loss with VPT Trained")


plt.figure(figsize=(10, 8))
img, label = trainset[i]
unscaled_img = img * clip_std + clip_mean

plt.subplot(1, 2, 1)
plt.imshow(np.transpose(unscaled_img, (1, 2, 0)))
plt.title(f"{classes[label]} before adding VP")
plt.axis("off")

plt.subplot(1, 2, 2)
img = img.to(device) + model4.learned_bias
unscaled_img = img * clip_std + clip_mean
plt.imshow(np.transpose(unscaled_img.cpu().detach().numpy(), (1, 2, 0)))
plt.title(f"{classes[label]} after adding VP")
plt.axis("off")


y_pred, y_true = get_predictions(model4, valloader, text_features, device)
scores = evaluate(y_pred, y_true, classes)


scores.round(4)


test_model1 = CLIPModel.from_pretrained(model_name).to(device)


test_model2 = CLIPModel.from_pretrained(model_name).to(device)
test_model2 = get_peft_model(test_model2, lora_config)

optimizer = torch.optim.SGD(
    test_model2.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
train(
    test_model2,
    trainvalloader,
    testloader,
    criterion_wrapper,
    optimizer,
    epochs,
    device,
    report_val=False,
)


test_model3 = CLIPModel.from_pretrained(model_name).to(device)
test_model3 = get_peft_model(test_model3, lora_config)

optimizer = torch.optim.SGD(
    test_model3.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
criterion = tecoa_loss
train(
    test_model3,
    trainvalloader,
    testloader,
    criterion,
    optimizer,
    epochs,
    device,
    report_val=False,
)


test_model4 = VPT_CLIP(224).to(device)

optimizer = torch.optim.SGD(
    test_model4.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
criterion = tecoa_loss
train(
    test_model4,
    trainvalloader,
    testloader,
    criterion,
    optimizer,
    epochs,
    device,
    report_val=False,
)


models = [test_model1, test_model2, test_model3, test_model4]
preds = []
trues = []

testloader.set_clean_mode()
for model in models:
    y_pred, y_true = get_predictions(model, testloader, text_features, device)
    preds.append(y_pred)
    trues.append(y_true)

testloader.set_attack_mode()
for model in models:
    y_pred, y_true = get_predictions(model, testloader, text_features, device)
    preds.append(y_pred)
    trues.append(y_true)


names = [
    "Clean CLIP",
    "Clean LORA Crossentropy Loss",
    "Clean LORA TeCoA Loss",
    "Clean VPT TeCoA Loss",
    "Adv CLIP",
    "Adv LORA Crossentropy Loss",
    "Adv LORA TeCoA Loss",
    "Adv VPT TeCoA Loss",
]
scores = defaultdict(lambda: defaultdict(float))

for i, name in enumerate(names):
    true = trues[i]
    pred = preds[i]

    acc = accuracy_score(true, pred)
    precision = precision_score(true, pred, average="weighted")
    recall = recall_score(true, pred, average="weighted")
    f_score = f1_score(true, pred, average="weighted")

    scores[name]["Accuracy"] = acc
    scores[name]["Precision"] = precision
    scores[name]["Recall"] = recall
    scores[name]["F1 Score"] = f_score

pd.DataFrame(scores).T.round(4)


df = pd.DataFrame(scores).T

df_long = df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
df_long.rename(columns={"index": "Model"}, inplace=True)

df_clean = df_long[df_long["Model"].str.startswith("Clean")]
df_adv = df_long[df_long["Model"].str.startswith("Adv")]

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharey=True)

sns.barplot(data=df_clean, x="Metric", y="Score", hue="Model", ax=axes[0])
axes[0].set_title("Clean Models")

sns.barplot(data=df_adv, x="Metric", y="Score", hue="Model", ax=axes[1])
axes[1].set_title("Adversarial Models")

for ax in axes:
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")

fig.tight_layout()
plt.show()


class AdversarialDataLoader(DataLoader):
    def __init__(self, *args, attack, clean_args, adv_args, device, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack = attack
        self.device = device
        self.attack_mode = False
        self.clean_mode = False

        self.clean_mean = clean_args["mean"].to(device)
        self.clean_std = clean_args["std"].to(device)
        self.clean_size = clean_args["size"]

        self.adv_mean = adv_args["mean"].to(device)
        self.adv_std = adv_args["std"].to(device)
        self.adv_size = adv_args["size"]

    def _apply_attack(self, images, labels):
        images = images * self.clean_std + self.clean_mean
        images = F.interpolate(
            images, size=self.adv_size, mode="bilinear", align_corners=False
        )
        images = (images - self.adv_mean) / self.adv_std

        images = self.attack(images, labels)

        images = images * self.adv_std + self.adv_mean
        images = F.interpolate(
            images, size=self.clean_size, mode="bilinear", align_corners=False
        )
        images = (images - self.clean_mean) / self.clean_std

        return images

    def __iter__(self):
        data_iter = super().__iter__()
        for images, labels in data_iter:
            images, labels = images.to(self.device), labels.to(self.device)

            if self.attack_mode and not self.clean_mode:
                images = self._apply_attack(images, labels)
                yield images, labels
            elif self.attack_mode:
                adv_images = self._apply_attack(images, labels)
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


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import Subset, random_split, DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.metrics import precision_recall_fscore_support as score
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from collections import defaultdict
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import torchattacks
import seaborn as sns
import pandas as pd

np.random.seed(42)


classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

cifar_mean = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(3, 1, 1)
cifar_std = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(3, 1, 1)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(clip_mean, clip_std),
    ]
)


num_train = 10000
train_size = int(0.8 * num_train)
val_size = num_train - train_size
batch_size = 64

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
indices = np.random.choice(len(trainset), num_train)
trainvalset = Subset(trainset, indices)
train_subset, val_subset = random_split(trainvalset, [train_size, val_size])
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

len(train_subset), len(testset)


n_sample = 5

plt.figure(figsize=(10, 8))
for i in range(n_sample):
    img, label = trainset[i]
    img = img * clip_std + clip_mean
    plt.subplot(1, n_sample, i + 1)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.title(classes[label])
    plt.axis("off")


target_model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
)
target_model = target_model.to(device)
target_model.eval()
print()


model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()
processor = CLIPProcessor.from_pretrained(model_name)


class_texts = [f"a photo of a {c}" for c in classes]
inputs = processor(text=class_texts, return_tensors="pt", padding=True).to(device)
text_features = model.get_text_features(**inputs).to(device)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
text_features = text_features.detach()


class AdversarialDataLoader(DataLoader):
    def __init__(self, *args, attack, clean_args, adv_args, device, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack = attack
        self.device = device
        self.attack_mode = False
        self.clean_mode = False

        self.clean_mean = clean_args["mean"].to(device)
        self.clean_std = clean_args["std"].to(device)
        self.clean_size = clean_args["size"]

        self.adv_mean = adv_args["mean"].to(device)
        self.adv_std = adv_args["std"].to(device)
        self.adv_size = adv_args["size"]

    def _apply_attack(self, images, labels):
        images = images * self.clean_std + self.clean_mean
        images = F.interpolate(
            images, size=self.adv_size, mode="bilinear", align_corners=False
        )
        images = (images - self.adv_mean) / self.adv_std

        images = self.attack(images, labels)

        images = images * self.adv_std + self.adv_mean
        images = F.interpolate(
            images, size=self.clean_size, mode="bilinear", align_corners=False
        )
        images = (images - self.clean_mean) / self.clean_std

        return images

    def __iter__(self):
        data_iter = super().__iter__()
        for images, labels in data_iter:
            images, labels = images.to(self.device), labels.to(self.device)

            if self.attack_mode and not self.clean_mode:
                images = self._apply_attack(images, labels)
                yield images, labels
            elif self.attack_mode:
                adv_images = self._apply_attack(images, labels)
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


attack = torchattacks.PGD(
    target_model, eps=8 / 255, alpha=2 / 255, steps=7, random_start=True
)
attack.set_normalization_used(mean=cifar_mean, std=cifar_std)
clean_args = {"mean": clip_mean, "std": clip_std, "size": 224}
adv_args = {"mean": cifar_mean, "std": cifar_std, "size": 32}


trainvalloader = AdversarialDataLoader(
    trainvalset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    attack=attack,
    clean_args=clean_args,
    adv_args=adv_args,
    device=device,
)
trainloader = AdversarialDataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    attack=attack,
    clean_args=clean_args,
    adv_args=adv_args,
    device=device,
)
valloader = AdversarialDataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    clean_args=clean_args,
    adv_args=adv_args,
    device=device,
)
testloader = AdversarialDataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    attack=attack,
    clean_args=clean_args,
    adv_args=adv_args,
    device=device,
)


def get_predictions(model, data_loader, text_features, device):
    y_pred = []
    y_true = []
    model = model.eval().to(device)

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = torch.matmul(image_features, text_features.T)
            probs = logits.softmax(dim=-1)
            y_pred.append(logits.argmax(dim=-1).cpu().numpy())

        y_true.append(labels.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0).flatten()
    y_pred = np.concatenate(y_pred, axis=0).flatten()
    return y_pred, y_true
