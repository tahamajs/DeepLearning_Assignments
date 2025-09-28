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
    """Comprehensive evaluation for medical imaging classification."""
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    if y_pred_proba is not None and len(np.unique(y_true)) > 2:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_true_bin.shape[1]

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auc_score = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc_score:.3f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-class ROC Curves")
        plt.legend()
        plt.grid(True)
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


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    auc,
)
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model


class CONFIG:
    seed = 42
    width, height = 150, 150
    train_path = "data/Data/train"
    test_path = "data/Data/test"
    aug_factor = 5
    num_classes = 3
    classes = ["NORMAL", "PNEUMONIA", "COVID19"]

    input_dimension = (150, 150, 3)
    filter_to_learn = (64, 64, 128, 128, 256, 256)
    max_pooling = (2, 2)
    batch_normalization_axis = -1
    cnn_activation_function = "relu"
    fcn_number_of_neurons = (512, 256)
    fcn_activation_function = ("relu", "relu")
    fcn_output_activation = "softmax"
    dropout_rate = 0.2
    kernel_size = (3, 3)
    number_of_cnn_layers = 6
    number_of_fcn_layers = 3

    epochs = 100
    batch_size = 32
    loss_function = "sparse_categorical_crossentropy"
    val_size = 0.35
    patience = 25
    start_from_epoch = 40


get_ipython().system("pip install kaggle")


get_ipython().system(" mkdir ~/.kaggle")


get_ipython().system(" chmod 600 ~/.kaggle/kaggle.json")


get_ipython().system(
    "kaggle datasets download -d prashant268/chest-xray-covid19-pneumonia -p ./data --unzip"
)


classes = [image_dir for image_dir in os.listdir(CONFIG.train_path)]
classes


train_dir = [CONFIG.train_path + "/" + c for c in classes]
test_dir = [CONFIG.test_path + "/" + c for c in classes]


file_formats = set()
for image_dir in train_dir:
    file_formats.update([file.split(".")[-1].lower() for file in os.listdir(image_dir)])
file_formats


n_classes = []
for image_dir in train_dir:
    n_classes.append(len(os.listdir(image_dir)))
    print("Number of files in %s: " % image_dir.split("/")[-1] + str(n_classes[-1]))
plt.bar(classes, n_classes)
plt.title("Distribution of train data")
plt.show()


classes = []
n_classes = []
for image_dir in test_dir:
    classes.append(image_dir.split("/")[-1].title())
    n_classes.append(len(os.listdir(image_dir)))
    print("Number of files in %s: " % image_dir.split("/")[-1] + str(n_classes[-1]))
plt.bar(classes, n_classes)
plt.title("Distribution of test data")
plt.show()


classes = []
n_classes = []
min_w = np.inf
min_h = np.inf
max_w, max_h = 0, 0
for image_dir in train_dir:
    for dir in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, dir))
        width, height, _ = image.shape
        min_w = min(min_w, width)
        min_h = min(min_h, height)
        max_w = max(max_w, width)
        max_h = max(max_h, height)
min_w, min_h, max_w, max_h


def load_balanced_data(directory, width, height):
    class_dict = {}
    balanced_images = []
    balanced_labels = []
    labels = [image_dir for image_dir in os.listdir(directory)]
    directories = [directory + "/" + c for c in labels]
    n_labels = []
    for dir in directories:
        n_labels.append(len(os.listdir(dir)))
    n_images = min(n_labels)

    for i, dir in enumerate(directories):
        class_dict[labels[i]] = 0
        images = np.array(os.listdir(dir))
        np.random.shuffle(images)
        for img_file in images[:n_images]:
            image = cv2.imread(os.path.join(dir, img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height))
            balanced_images.append(image)
            balanced_labels.append(labels[i])
            class_dict[labels[i]] += 1

    balanced_images = np.array(balanced_images)
    balanced_labels = np.array(balanced_labels)
    indices = np.arange(len(balanced_images))
    np.random.shuffle(indices)
    balanced_images = balanced_images[indices]
    balanced_labels = balanced_labels[indices]
    return balanced_images, balanced_labels


X_train, y_train = load_balanced_data(CONFIG.train_path, CONFIG.width, CONFIG.height)
X_test, y_test = load_balanced_data(CONFIG.test_path, CONFIG.width, CONFIG.height)


num_to_label = {0: "NORMAL", 1: "PNEUMONIA", 2: "COVID19"}
label_to_num = {"NORMAL": 0, "PNEUMONIA": 1, "COVID19": 2}


def preprocess_data(X, y):
    X = X / 255
    y = np.array([label_to_num[label] for label in y])
    return X, y


X_train, y_train = preprocess_data(X_train, y_train)
X_test, y_test = preprocess_data(X_test, y_test)


X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=CONFIG.val_size,
    random_state=CONFIG.seed,
    stratify=y_train,
)


np.unique(y_train, return_counts=True)


X_train.shape


i = 0
for image, label in zip(X_train[:5], y_train[:5]):
    plt.subplot(1, 5, i + 1)
    i += 1
    plt.imshow(image)
    plt.title(num_to_label[label])
    plt.axis("off")


data_augmentation = tf.keras.Sequential(
    [
        layers.Input((CONFIG.height, CONFIG.width, 3)),
        layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="constant", fill_value=0
        ),
        layers.RandomRotation(0.05, fill_mode="constant", fill_value=0),
    ]
)


plt.figure(figsize=(10, 10))
image = X_train[0]
plt.subplot(3, 3, 1)
plt.imshow(image)
plt.title("Before augmentaion")
plt.axis("off")
for i in range(1, 6):
    plt.subplot(3, 3, i + 1)
    plt.imshow(tf.squeeze(data_augmentation(tf.expand_dims(image, axis=0))))
    plt.title("After augmentaion")
    plt.axis("off")
plt.show()


def augment_data(images, labels, aug_factor, augmentation):
    augmented_images = []
    augmented_images.extend(images)
    augmented_labels = np.tile(labels, aug_factor)

    for _ in range(aug_factor - 1):
        augmented_images.extend(augmentation(images))

    indices = np.arange(len(augmented_labels))
    np.random.shuffle(indices)
    augmented_images = np.array(augmented_images)[indices]
    augmented_labels = np.array(augmented_labels)[indices]
    return augmented_images, augmented_labels


X_train_aug, y_train_aug = augment_data(
    X_train, y_train, CONFIG.aug_factor, data_augmentation
)


X_train_aug.shape


i = 0
for image, label in zip(X_train_aug[:5], y_train[:5]):
    plt.subplot(1, 5, i + 1)
    i += 1
    plt.imshow(image)
    plt.title(num_to_label[label])
    plt.axis("off")


model = create_model()
model.summary()


def plot_history(history, lr):
    lr = lr if type(lr) == float else lr.name
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"accuracy of model with lr={lr}")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"loss of model with lr={lr}")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


def evaluate(predicted_proba, actual_values):
    classes = CONFIG.classes
    predictions = np.argmax(predicted_proba, axis=1)
    precision, recall, fscore, support = score(actual_values, predictions)

    def print_foreach_class(metric):
        for i, c in enumerate(classes):
            print(f"\t{c}: {(metric[i]):.2%}")

    print(f"Accuracy: {accuracy_score(actual_values,predictions):.2%}")
    print(f"Accuracy for each class:")
    matrix = confusion_matrix(actual_values, predictions)
    print_foreach_class(matrix.diagonal() / matrix.sum(axis=1))

    print(f"Precision:")
    print_foreach_class(precision)
    print(f"\tMicro: {precision_score(actual_values,predictions,average='micro'):.2%}")
    print(f"\tMacro: {precision_score(actual_values,predictions,average='macro'):.2%}")

    print(f"Recall:")
    print_foreach_class(recall)
    print(f"\tMicro: {recall_score(actual_values,predictions,average='micro'):.2%}")
    print(f"\tMacro: {recall_score(actual_values,predictions,average='macro'):.2%}")

    print(f"F1 score:")
    print_foreach_class(fscore)
    print(f"\tMicro: {f1_score(actual_values,predictions,average='micro'):.2%}")
    print(f"\tMacro: {f1_score(actual_values,predictions,average='macro'):.2%}")

    sns.heatmap(
        matrix,
        annot=True,
        cmap="flare",
        xticklabels=CONFIG.classes,
        yticklabels=CONFIG.classes,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()


def train_and_evaluate_model(model, learning_rate, X_train, y_train, X_val, y_val):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=CONFIG.loss_function, metrics=["accuracy"])
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=CONFIG.patience,
        restore_best_weights=True,
        start_from_epoch=CONFIG.start_from_epoch,
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=CONFIG.epochs,
        batch_size=CONFIG.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=2,
    )
    plot_history(history, learning_rate)
    evaluate(model.predict(X_val), y_val)


print("Count:")
print(
    "train:",
    len(y_train),
    "augmented train:",
    len(y_train_aug),
    "validation:",
    len(y_val),
    "test:",
    len(y_test),
)


model = create_model()
train_and_evaluate_model(model, 0.001, X_train, y_train, X_val, y_val)


model = create_model()
train_and_evaluate_model(model, 0.0001, X_train_aug, y_train_aug, X_val, y_val)


model = create_model()
train_and_evaluate_model(model, 0.001, X_train_aug, y_train_aug, X_val, y_val)


model = create_model()
train_and_evaluate_model(model, 0.01, X_train_aug, y_train_aug, X_val, y_val)


model = create_model()
train_and_evaluate_model(model, 0.1, X_train_aug, y_train_aug, X_val, y_val)


initial_lr = 1e-2
decay_steps = 50


def plot_scheduler(step, schedulers):
    plt.title(schedulers[0][0].name)
    for scheduler, label in schedulers:
        plt.plot(range(step), scheduler(range(step)), label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
    plt.legend()
    plt.show()


cosine_decays = []
cosine_decays.append(
    (keras.optimizers.schedules.CosineDecay(initial_lr, 100, alpha=0), "alpha=0")
)
cosine_decays.append(
    (
        keras.optimizers.schedules.CosineDecay(initial_lr, 100, alpha=0.0001),
        "alpha=0.0001",
    )
)
cosine_decays.append(
    (keras.optimizers.schedules.CosineDecay(initial_lr, 100, alpha=0.01), "alpha=0.01")
)
plot_scheduler(CONFIG.epochs, cosine_decays)


lr = keras.optimizers.schedules.CosineDecay(initial_lr, 100, alpha=0)
model = create_model()
train_and_evaluate_model(model, lr, X_train_aug, y_train_aug, X_val, y_val)


exponential_decays = []
exponential_decays.append(
    (
        keras.optimizers.schedules.ExponentialDecay(
            initial_lr,
            decay_steps=decay_steps,
            decay_rate=0.1,
        ),
        "decay_rate=0.1",
    )
)
exponential_decays.append(
    (
        keras.optimizers.schedules.ExponentialDecay(
            initial_lr,
            decay_steps=decay_steps,
            decay_rate=0.5,
        ),
        "decay_rate=0.5",
    )
)
exponential_decays.append(
    (
        keras.optimizers.schedules.ExponentialDecay(
            initial_lr,
            decay_steps=decay_steps,
            decay_rate=0.9,
        ),
        "decay_rate=0.9",
    )
)
plot_scheduler(CONFIG.epochs, exponential_decays)


lr = keras.optimizers.schedules.ExponentialDecay(
    initial_lr, decay_steps, decay_rate=0.9
)
model = create_model()
train_and_evaluate_model(model, lr, X_train_aug, y_train_aug, X_val, y_val)


cosine_restart_schedules = []
cosine_restart_schedules.append(
    (
        tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_lr,
            first_decay_steps=decay_steps // 4,
            t_mul=1.0,
            m_mul=1.0,
            alpha=0.0001,
        ),
        "alpha=0.0001 m_mul=1 t_mul=1",
    )
)
cosine_restart_schedules.append(
    (
        tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_lr,
            first_decay_steps=decay_steps // 4,
            t_mul=1.0,
            m_mul=1.0,
            alpha=0.9,
        ),
        "alpha=0.9 m_mul=1 t_mul=1",
    )
)
cosine_restart_schedules.append(
    (
        tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_lr,
            first_decay_steps=decay_steps // 4,
            t_mul=2.0,
            m_mul=0.5,
            alpha=0.0001,
        ),
        "alpha=0.0001 m_mul=0.5 t_mul=2",
    )
)

plot_scheduler(CONFIG.epochs, cosine_restart_schedules)


lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_lr, first_decay_steps=decay_steps // 4, t_mul=2.0, m_mul=0.5, alpha=0.0001
)
model = create_model()
train_and_evaluate_model(model, lr, X_train_aug, y_train_aug, X_val, y_val)


def augment_data_like_paper(images, labels):
    augmented_labels = np.tile(labels, 5)
    augmented_images = []
    augmented_images.extend(images)
    augmented_images.extend(images[:, :, ::-1, :])
    rot90_images = np.rot90(images, axes=(1, 2))
    augmented_images.extend(rot90_images)
    rot180_images = np.rot90(rot90_images, axes=(1, 2))
    augmented_images.extend(rot180_images)
    augmented_images.extend(np.rot90(rot180_images, axes=(1, 2)))

    indices = np.arange(len(augmented_labels))
    np.random.shuffle(indices)
    augmented_images = np.array(augmented_images)[indices]
    augmented_labels = np.array(augmented_labels)[indices]
    return augmented_images, augmented_labels


X_train_aug_paper, y_train_aug_paper = augment_data_like_paper(X_train, y_train)


X_train_aug_paper.shape, y_train_aug_paper.shape


plt.figure(figsize=(10, 10))
image, label = X_train[1:2], y_train[1:2]
aug_images, _ = augment_data_like_paper(image, label)
plt.subplot(3, 3, 1)
plt.imshow(image[0])
plt.title("Before augmentaion")
plt.axis("off")
for i in range(aug_images.shape[0]):
    plt.subplot(3, 3, i + 2)
    plt.imshow(aug_images[i])
    plt.title("After augmentaion")
    plt.axis("off")
plt.show()


i = 0
for image, label in zip(X_train_aug_paper[:10], y_train_aug_paper[:10]):
    plt.subplot(2, 5, i + 1)
    i += 1
    plt.imshow(image)
    plt.title(num_to_label[label])
    plt.axis("off")


model = create_model()
train_and_evaluate_model(
    model, 0.01, X_train_aug_paper, y_train_aug_paper, X_val, y_val
)


def load_balanced_gray_data(directory, width, height):
    class_dict = {}
    balanced_images = []
    balanced_labels = []
    labels = [image_dir for image_dir in os.listdir(directory)]
    directories = [directory + "/" + c for c in labels]
    n_labels = []
    for dir in directories:
        n_labels.append(len(os.listdir(dir)))
    n_images = min(n_labels)

    for i, dir in enumerate(directories):
        class_dict[labels[i]] = 0
        images = np.array(os.listdir(dir))
        np.random.shuffle(images)
        for img_file in images[:n_images]:
            image = cv2.imread(os.path.join(dir, img_file), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (width, height))
            balanced_images.append(image)
            balanced_labels.append(labels[i])
            class_dict[labels[i]] += 1

    balanced_images = np.array(balanced_images)
    balanced_labels = np.array(balanced_labels)
    indices = np.arange(len(balanced_images))
    np.random.shuffle(indices)
    balanced_images = balanced_images[indices]
    balanced_labels = balanced_labels[indices]
    return balanced_images, balanced_labels


X_train, y_train = load_balanced_gray_data(
    CONFIG.train_path, CONFIG.width, CONFIG.height
)


X_train, y_train = preprocess_data(X_train, y_train)


X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=CONFIG.val_size,
    random_state=CONFIG.seed,
    stratify=y_train,
)


i = 0
for image, label in zip(X_train[:5], y_train[:5]):
    plt.subplot(1, 5, i + 1)
    i += 1
    plt.imshow(image, cmap="gray")
    plt.title(num_to_label[label])
    plt.axis("off")


data_augmentation_gray = tf.keras.Sequential(
    [
        layers.Input((CONFIG.height, CONFIG.width, 1)),
        layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="constant", fill_value=0
        ),
        layers.RandomRotation(0.05, fill_mode="constant", fill_value=0),
    ]
)


X_train = np.expand_dims(X_train, axis=3)
X_val = np.expand_dims(X_val, axis=3)


X_train_aug, y_train_aug = augment_data(
    X_train, y_train, CONFIG.aug_factor, data_augmentation_gray
)


model = tf.keras.Sequential(
    [
        layers.Input((CONFIG.height, CONFIG.width, 1)),
    ]
)
for i in range(CONFIG.number_of_cnn_layers):
    model.add(
        layers.Conv2D(
            filters=CONFIG.filter_to_learn[i],
            kernel_size=CONFIG.kernel_size,
            padding="same",
        )
    )
    model.add(layers.MaxPool2D(pool_size=CONFIG.max_pooling))
    model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
    model.add(layers.ReLU())
    model.add(layers.Dropout(CONFIG.dropout_rate))

model.add(layers.Flatten())
for i in range(CONFIG.number_of_fcn_layers - 1):
    model.add(layers.Dense(CONFIG.fcn_number_of_neurons[i]))
    model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
    model.add(layers.ReLU())

model.add(layers.Dense(CONFIG.num_classes, activation=CONFIG.fcn_output_activation))


train_and_evaluate_model(model, 0.01, X_train_aug, y_train_aug, X_val, y_val)


model = tf.keras.Sequential(
    [
        layers.Input((CONFIG.height, CONFIG.width, 1)),
    ]
)
for i in range(CONFIG.number_of_cnn_layers):
    model.add(
        layers.Conv2D(
            filters=CONFIG.filter_to_learn[i] * 2,
            kernel_size=CONFIG.kernel_size,
            padding="same",
        )
    )
    model.add(layers.MaxPool2D(pool_size=CONFIG.max_pooling))
    model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
    model.add(layers.ReLU())
    model.add(layers.Dropout(CONFIG.dropout_rate))

model.add(layers.Flatten())
for i in range(CONFIG.number_of_fcn_layers - 1):
    model.add(layers.Dense(CONFIG.fcn_number_of_neurons[i]))
    model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
    model.add(layers.ReLU())

model.add(layers.Dense(128))
model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
model.add(layers.ReLU())

model.add(layers.Dense(64))
model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
model.add(layers.ReLU())

model.add(layers.Dense(32))
model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
model.add(layers.ReLU())

model.add(layers.Dense(CONFIG.num_classes, activation=CONFIG.fcn_output_activation))


train_and_evaluate_model(model, 0.01, X_train_aug, y_train_aug, X_val, y_val)


model = tf.keras.Sequential(
    [
        layers.Input((CONFIG.height, CONFIG.width, 1)),
    ]
)
for i in range(CONFIG.number_of_cnn_layers):
    model.add(
        layers.Conv2D(
            filters=CONFIG.filter_to_learn[i],
            kernel_size=CONFIG.kernel_size,
            padding="same",
        )
    )
    model.add(layers.MaxPool2D(pool_size=CONFIG.max_pooling))
    model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
    model.add(layers.ReLU())
    model.add(layers.Dropout(CONFIG.dropout_rate))

for i in range(2):
    model.add(
        layers.Conv2DTranspose(
            filters=CONFIG.filter_to_learn[CONFIG.number_of_cnn_layers - 1 - i],
            kernel_size=CONFIG.kernel_size,
            padding="same",
            strides=(2, 2),
        )
    )
    model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
    model.add(layers.ReLU())
    model.add(layers.Dropout(CONFIG.dropout_rate))

model.add(layers.Flatten())
for i in range(CONFIG.number_of_fcn_layers - 1):
    model.add(layers.Dense(CONFIG.fcn_number_of_neurons[i]))
    model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
    model.add(layers.ReLU())

model.add(layers.Dense(CONFIG.num_classes, activation=CONFIG.fcn_output_activation))


train_and_evaluate_model(model, 0.01, X_train_aug, y_train_aug, X_val, y_val)


X_train, y_train = load_balanced_gray_data(CONFIG.train_path, 300, 300)
X_train, y_train = preprocess_data(X_train, y_train)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=CONFIG.val_size,
    random_state=CONFIG.seed,
    stratify=y_train,
)


X_train = np.expand_dims(X_train, axis=3)
X_val = np.expand_dims(X_val, axis=3)


model = tf.keras.Sequential(
    [
        layers.Input((300, 300, 1)),
    ]
)
for i in range(CONFIG.number_of_cnn_layers):
    model.add(
        layers.Conv2D(filters=CONFIG.filter_to_learn[i], kernel_size=CONFIG.kernel_size)
    )
    model.add(layers.MaxPool2D(pool_size=CONFIG.max_pooling))
    model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
    model.add(layers.ReLU())
    model.add(layers.Dropout(CONFIG.dropout_rate))

model.add(layers.Flatten())
for i in range(CONFIG.number_of_fcn_layers - 1):
    model.add(layers.Dense(CONFIG.fcn_number_of_neurons[i]))
    model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
    model.add(layers.ReLU())

model.add(layers.Dense(CONFIG.num_classes, activation=CONFIG.fcn_output_activation))
train_and_evaluate_model(model, 0.01, X_train, y_train, X_val, y_val)


X_train, y_train = load_balanced_gray_data(
    CONFIG.train_path, CONFIG.width, CONFIG.height
)
X_test, y_test = load_balanced_gray_data(CONFIG.test_path, CONFIG.width, CONFIG.height)


X_train, y_train = preprocess_data(X_train, y_train)
X_test, y_test = preprocess_data(X_test, y_test)


X_train = np.expand_dims(X_train, axis=3)
X_train_aug, y_train_aug = augment_data(
    X_train, y_train, CONFIG.aug_factor, data_augmentation_gray
)


X_train_aug.shape


model = tf.keras.Sequential(
    [
        layers.Input((CONFIG.height, CONFIG.width, 1)),
    ]
)
for i in range(CONFIG.number_of_cnn_layers):
    model.add(
        layers.Conv2D(
            filters=CONFIG.filter_to_learn[i],
            kernel_size=CONFIG.kernel_size,
            padding="same",
        )
    )
    model.add(layers.MaxPool2D(pool_size=CONFIG.max_pooling))
    model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
    model.add(layers.ReLU())
    model.add(layers.Dropout(CONFIG.dropout_rate))

model.add(layers.Flatten())
for i in range(CONFIG.number_of_fcn_layers - 1):
    model.add(layers.Dense(CONFIG.fcn_number_of_neurons[i]))
    model.add(layers.BatchNormalization(axis=CONFIG.batch_normalization_axis))
    model.add(layers.ReLU())

model.add(layers.Dense(CONFIG.num_classes, activation=CONFIG.fcn_output_activation))
train_and_evaluate_model(model, 0.01, X_train_aug, y_train_aug, X_test, y_test)


def add_feedforward_layers(model, units):
    x = model.output
    x = layers.Flatten()(x)

    x = layers.Dense(units)(x)
    x = layers.BatchNormalization(axis=CONFIG.batch_normalization_axis)(x)
    x = layers.ReLU()(x)

    output = layers.Dense(CONFIG.num_classes, activation=CONFIG.fcn_output_activation)(
        x
    )

    return tf.keras.Model(inputs=model.input, outputs=output)


X_train, y_train = load_balanced_data(CONFIG.train_path, 224, 224)
X_test, y_test = load_balanced_data(CONFIG.test_path, 224, 224)


X_train.shape, X_test.shape


X_train, y_train = preprocess_data(X_train, y_train)
X_test, y_test = preprocess_data(X_test, y_test)


vgg16_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
vgg16_model.trainable = False


model = add_feedforward_layers(vgg16_model, 64)
model.summary()


train_and_evaluate_model(model, 0.01, X_train, y_train, X_test, y_test)


mobilenet_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
mobilenet_model.trainable = False


model = add_feedforward_layers(mobilenet_model, 16)
model.summary()


train_and_evaluate_model(model, 0.01, X_train, y_train, X_test, y_test)
