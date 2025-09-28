import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_iou(y_true, y_pred, num_classes):
    """Calculate IoU for each class and mean IoU."""
    ious = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
        else:
            iou = 0.0
        ious.append(iou)
    return np.array(ious), np.mean(ious)


def pixel_accuracy(y_true, y_pred):
    """Calculate pixel-level accuracy."""
    return np.mean(y_true == y_pred)


def evaluate_segmentation(y_true, y_pred, class_names=None, num_classes=None):
    """Comprehensive segmentation evaluation."""
    if num_classes is None:
        num_classes = len(np.unique(y_true))

    ious, mean_iou = calculate_iou(y_true, y_pred, num_classes)
    acc = pixel_accuracy(y_true, y_pred)

    print(f"Pixel Accuracy: {acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")

    if class_names:
        for i, (iou, name) in enumerate(zip(ious, class_names)):
            print(f"{name}: IoU = {iou:.4f}")

    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Pixel-level Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return {"pixel_accuracy": acc, "mean_iou": mean_iou, "ious": ious}


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
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
import cv2
import os


from google.colab import drive

drive.mount("/content/drive/")
path = "/content/drive/MyDrive/Colab/NNDL/CA3/Part1/dataset/CamVid/"


get_ipython().system("git clone https://github.com/lih627/CamVid.git")
path = "./CamVid/"


def load_dataset(path, file):
    """
    Load images and masks from a text file containing paths.
    Args:
      path: Directory containing the dataset files.
      file: Text file with lines of 'image_path mask_path'.
    Returns:
      X: Array of images.
      y: Array of masks.
    """
    X = []
    y = []
    with open(path + file, "r") as file:
        for line in file:
            X_path, y_path = line.split()
            img = cv2.imread(os.path.join(path, X_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            img = cv2.imread(os.path.join(path, y_path), cv2.IMREAD_GRAYSCALE)
            y.append(img)
    return np.array(X), np.array(y)


X_train, y_train = load_dataset(path, "camvid_train.txt")
X_val, y_val = load_dataset(path, "camvid_val.txt")
X_test, y_test = load_dataset(path, "camvid_test.txt")


X_train.shape, y_train.shape, X_val.shape, X_test.shape


CAMVID_CLASSES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Sidewalk",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
    "Void",
]


CAMVID_CLASS_COLORS = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (0, 0, 192),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
    (0, 0, 0),
]


_, counts = np.unique(y_train, return_counts=True)


plt.figure(figsize=(10, 5))
label_counts = list(zip(CAMVID_CLASSES, counts))
plt.grid(axis="y", linestyle="--", zorder=0)
normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in CAMVID_CLASS_COLORS]
sns.barplot(
    x=CAMVID_CLASSES, y=counts, hue=CAMVID_CLASSES, palette=normalized_colors, zorder=3
)
plt.xlabel("Class Labels")
plt.ylabel("Number of pixels in all training images")
plt.title("Class Distribution in Dataset")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


def plot_mask(mask):
    """
    Convert a segmentation mask to RGB for visualization.
    Args:
      mask: 2D or 3D array of class indices.
    """
    if mask.ndim == 3:
        mask = mask.argmax(axis=-1)
    rgb_mask = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for class_index, color in enumerate(CAMVID_CLASS_COLORS):
        rgb_mask[mask == class_index] = color
    rgb_mask[mask == 255] = CAMVID_CLASS_COLORS[-1]
    plt.imshow(rgb_mask)
    plt.axis("off")


indices = np.arange(4)
plt.figure(figsize=(12, 5))
for i, idx in enumerate(indices):
    plt.subplot(2, 4, i + 1)
    plt.title("Image")
    plt.imshow(X_val[idx])
    plt.axis("off")
    plt.subplot(2, 4, i + 5)
    plt.title("Ground truth mask")
    plot_mask(y_val[idx])
    plt.axis("off")


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Compute Dice coefficient for multi-class segmentation.
    Args:
      y_true: Ground truth labels.
      y_pred: Predicted probabilities.
      smooth: Smoothing factor to avoid division by zero.
    Returns:
      Mean Dice coefficient across classes.
    """
    num_classes = y_pred.shape[-1]
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=num_classes, axis=-1)
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes, axis=-1)
    intersection = tf.reduce_sum(tf.multiply(y_true_one_hot, y_pred), axis=[1, 2])
    sum_true = tf.reduce_sum(y_true_one_hot, axis=[1, 2])
    sum_pred = tf.reduce_sum(y_pred, axis=[1, 2])
    return tf.reduce_mean(
        (2.0 * intersection + smooth) / (sum_true + sum_pred + smooth)
    )


def iou_score(y_true, y_pred, smooth=1e-6):
    """
    Compute Intersection over Union (IoU) for multi-class segmentation.
    Args:
      y_true: Ground truth labels.
      y_pred: Predicted probabilities.
      smooth: Smoothing factor.
    Returns:
      Mean IoU across classes.
    """
    num_classes = y_pred.shape[-1]
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=num_classes, axis=-1)
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes, axis=-1)
    intersection = tf.reduce_sum(tf.multiply(y_true_one_hot, y_pred), axis=[1, 2])
    sum_true = tf.reduce_sum(y_true_one_hot, axis=[1, 2])
    sum_pred = tf.reduce_sum(y_pred, axis=[1, 2])
    union = sum_true + sum_pred - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


def dice_loss(y_true, y_pred, smooth=1e-6):
    num_classes = y_pred.shape[-1]
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes, axis=-1)
    intersection = tf.reduce_sum(tf.multiply(y_true_one_hot, y_pred), axis=[1, 2])
    sum_true = tf.reduce_sum(y_true_one_hot, axis=[1, 2])
    sum_pred = tf.reduce_sum(y_pred, axis=[1, 2])
    return 1 - tf.reduce_mean(
        (2.0 * intersection + smooth) / (sum_true + sum_pred + smooth)
    )


def iou_loss(y_true, y_pred, smooth=1e-6):
    num_classes = y_pred.shape[-1]
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes, axis=-1)
    intersection = tf.reduce_sum(tf.multiply(y_true_one_hot, y_pred), axis=[1, 2])
    sum_true = tf.reduce_sum(y_true_one_hot, axis=[1, 2])
    sum_pred = tf.reduce_sum(y_pred, axis=[1, 2])
    union = sum_true + sum_pred - intersection
    return 1 - tf.reduce_mean((intersection + smooth) / (union + smooth))


class CONFIG:
    seed = 42
    height, width = 720, 960
    num_classes = 11

    t = [None, None, None, 6, 6, 6, None, None, None, None]
    c = [32, 48, 64, 64, 96, 128, 128, 128, 128, None]
    n = [1, 1, 1, 3, 3, 3, None, None, 2, 1]
    s = [2, 2, 2, 2, 2, 1, None, None, 1, 1]

    epochs = 100
    batch_size = 16
    input_shape = (720, 960, 3)
    loss_function = keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, ignore_class=255
    )
    optimizer = keras.optimizers.SGD
    scheduler = keras.optimizers.schedules.PolynomialDecay(
        0.045,
        1000,
        end_learning_rate=0.0001,
        power=0.9,
        cycle=False,
        name="PolynomialDecay",
    )
    regularizer = keras.regularizers.L2(0.00004)


class DSConv(layers.Layer):
    """
    Depthwise Separable Convolution layer with batch norm and ReLU.
    """

    def __init__(self, channels, strides):
        super(DSConv, self).__init__()
        self.conv = layers.SeparableConv2D(
            channels,
            kernel_size=3,
            strides=strides,
            padding="same",
            depth_multiplier=1,
            use_bias=False,
            pointwise_regularizer=CONFIG.regularizer,
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class Bottleneck(layers.Layer):
    """
    Bottleneck block with expansion, depthwise conv, and residual connection.
    """

    def __init__(self, expansion_factor, in_channels, out_channels, strides, repeat):
        super(Bottleneck, self).__init__()
        self.blocks = []

        for i in range(repeat):
            if i == 0:
                self.blocks.append(
                    self._make_block(
                        expansion_factor, in_channels, out_channels, strides
                    )
                )
            else:
                self.blocks.append(
                    self._make_block(expansion_factor, out_channels, out_channels, 1)
                )

    def _make_block(self, expansion_factor, in_channels, out_channels, strides):
        model = models.Sequential(
            [
                layers.Conv2D(
                    expansion_factor * in_channels,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=CONFIG.regularizer,
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.DepthwiseConv2D(
                    kernel_size=3, strides=strides, padding="same", use_bias=False
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    out_channels,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=CONFIG.regularizer,
                ),
                layers.BatchNormalization(),
            ]
        )
        return model

    def call(self, x, training=False):
        out = x
        for block in self.blocks:
            residue = out
            out = block(out, training=training)
            if residue.shape == out.shape:
                out = layers.Add()([out, residue])
        return out


class PPM(layers.Layer):
    """
    Pyramid Pooling Module for multi-scale context aggregation.
    """

    def __init__(self, input_shape, out_channels, pool_sizes=[1, 2, 3, 6]):
        super(PPM, self).__init__()
        self.pool_modules = []
        out_channels_per_pool = out_channels // len(pool_sizes)
        h, w = input_shape[:2]

        for pool_size in pool_sizes:
            self.pool_modules.append(
                models.Sequential(
                    [
                        layers.AveragePooling2D(
                            pool_size=(h // pool_size, w // pool_size),
                            strides=(h // pool_size, w // pool_size),
                            padding="valid",
                        ),
                        layers.Conv2D(
                            out_channels_per_pool,
                            kernel_size=1,
                            use_bias=False,
                            kernel_regularizer=CONFIG.regularizer,
                        ),
                        layers.BatchNormalization(),
                        layers.ReLU(),
                    ]
                )
            )
        self.conv = layers.Conv2D(
            out_channels,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=CONFIG.regularizer,
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        h, w = x.shape[1:3]
        pools = [x]
        for pool_module in self.pool_modules:
            pooled = pool_module(x, training=training)
            features = tf.image.resize(pooled, (h, w), method="bilinear")
            pools.append(features)
        out = tf.concat(pools, axis=-1)
        out = self.conv(out)
        out = self.bn(out, training=training)
        out = self.relu(out)
        return out


class FFM(layers.Layer):
    """
    Feature Fusion Module to combine high and low resolution features.
    """

    def __init__(self, high_channels, low_channels, out_channels, dilation=4):
        super(FFM, self).__init__()
        self.high_conv = models.Sequential(
            [
                layers.Conv2D(
                    out_channels,
                    kernel_size=1,
                    use_bias=False,
                    kernel_regularizer=CONFIG.regularizer,
                ),
                layers.BatchNormalization(),
            ]
        )
        self.low_dwconv = models.Sequential(
            [
                layers.DepthwiseConv2D(
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    dilation_rate=dilation,
                    use_bias=False,
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )
        self.low_conv = models.Sequential(
            [
                layers.Conv2D(
                    out_channels,
                    kernel_size=1,
                    use_bias=False,
                    kernel_regularizer=CONFIG.regularizer,
                ),
                layers.BatchNormalization(),
            ]
        )
        self.relu = layers.ReLU()

    def call(self, high_res, low_res, training=False):
        low_res = tf.image.resize(low_res, high_res.shape[1:3], method="bilinear")
        low_res = self.low_dwconv(low_res, training=training)
        low_res = self.low_conv(low_res, training=training)
        high_res = self.high_conv(high_res, training=training)
        out = layers.Add()([high_res, low_res])
        out = self.relu(out)
        return out


class FastSCNN(models.Model):
    """
    Fast-SCNN model for real-time semantic segmentation.
    """

    def __init__(self, num_classes):
        super(FastSCNN, self).__init__()
        self.rescale = layers.Rescaling(scale=1.0 / 255)
        self.learning_to_downsample = models.Sequential(
            [
                layers.Conv2D(
                    CONFIG.c[0],
                    kernel_size=3,
                    strides=CONFIG.s[0],
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=CONFIG.regularizer,
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                DSConv(CONFIG.c[1], CONFIG.s[1]),
                DSConv(CONFIG.c[2], CONFIG.s[2]),
            ]
        )

        self.global_feature_extractor = models.Sequential(
            [
                Bottleneck(
                    CONFIG.t[3], CONFIG.c[2], CONFIG.c[3], CONFIG.s[3], CONFIG.n[3]
                ),
                Bottleneck(
                    CONFIG.t[4], CONFIG.c[3], CONFIG.c[4], CONFIG.s[4], CONFIG.n[4]
                ),
                Bottleneck(
                    CONFIG.t[5], CONFIG.c[4], CONFIG.c[5], CONFIG.s[5], CONFIG.n[5]
                ),
                PPM((23, 30, 128), CONFIG.c[6]),
            ]
        )

        self.feature_fusion_module = FFM(CONFIG.c[2], CONFIG.c[6], CONFIG.c[7])

        self.classifier = models.Sequential(
            [
                DSConv(CONFIG.c[8], CONFIG.s[8]),
                DSConv(CONFIG.c[8], CONFIG.s[8]),
                layers.Conv2D(
                    num_classes,
                    kernel_size=1,
                    strides=CONFIG.s[9],
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=CONFIG.regularizer,
                ),
                layers.Dropout(0.1),
                layers.Softmax(),
            ]
        )

    def call(self, x, training=False):
        x = self.rescale(x)
        high_res = self.learning_to_downsample(x, training=training)
        low_res = self.global_feature_extractor(high_res, training=training)
        out = self.feature_fusion_module(high_res, low_res, training=training)
        out = self.classifier(out, training=training)
        return tf.image.resize(out, x.shape[1:3], method="bilinear")


model = FastSCNN(CONFIG.num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(CONFIG.scheduler),
    loss=CONFIG.loss_function,
    metrics=["accuracy", dice_coefficient, iou_score],
)


hist = model.fit(
    X_train,
    y_train,
    batch_size=CONFIG.batch_size,
    epochs=100,
    verbose=2,
    validation_data=(X_val, y_val),
)


model.summary()


for layer in model.learning_to_downsample.layers:
    print(layer.name, layer.count_params())


for layer in model.global_feature_extractor.layers:
    print(layer.name, layer.count_params())


layer = model.feature_fusion_module
print(layer.name, layer.count_params())


for layer in model.classifier.layers:
    print(layer.name, layer.count_params())


def plot_history(hist):
    """
    Plot training history for loss and metrics.
    Args:
      hist: History object from model.fit().
    """
    for key, value in hist.history.items():
        if "val_" in key:
            continue
        if "loss" not in key:
            plt.ylim(0, 1)
        validation_value = hist.history["val_" + key]
        plt.title(f"Model {key}")
        plt.xlabel("Epochs")
        plt.ylabel(key)
        plt.plot(value, label="Train")
        plt.plot(validation_value, label="Validation")
        plt.legend()
        plt.show()


plot_history(hist)


indices = np.random.choice(X_val.shape[0], 3)
preds = model.predict(X_val[indices])
plt.figure(figsize=(10, 10))
for i, idx in enumerate(indices):
    plt.subplot(3, 3, i + 1)
    plt.title("Image")
    plt.imshow(X_val[idx])
    plt.axis("off")
    plt.subplot(3, 3, i + 4)
    plt.title("Ground truth mask")
    plot_mask(y_val[idx])
    plt.axis("off")
    plt.subplot(3, 3, i + 7)
    plt.title("Predicted mask")
    plot_mask(preds[i])
    plt.axis("off")


model = FastSCNN(CONFIG.num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(CONFIG.scheduler),
    loss=iou_loss,
    metrics=["accuracy", dice_coefficient, iou_score],
)


hist = model.fit(
    X_train,
    y_train,
    batch_size=CONFIG.batch_size,
    epochs=100,
    verbose=2,
    validation_data=(X_val, y_val),
)


plot_history(hist)


indices = np.random.choice(X_val.shape[0], 3)
preds = model.predict(X_val[indices])
plt.figure(figsize=(10, 10))
for i, idx in enumerate(indices):
    plt.subplot(3, 3, i + 1)
    plt.title("Image")
    plt.imshow(X_val[idx])
    plt.axis("off")
    plt.subplot(3, 3, i + 4)
    plt.title("Ground truth mask")
    plot_mask(y_val[idx])
    plt.axis("off")
    plt.subplot(3, 3, i + 7)
    plt.title("Predicted mask")
    plot_mask(preds[i])
    plt.axis("off")


model = FastSCNN(CONFIG.num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(CONFIG.scheduler),
    loss=dice_loss,
    metrics=["accuracy", dice_coefficient, iou_score],
)


hist = model.fit(
    X_train,
    y_train,
    batch_size=CONFIG.batch_size,
    epochs=100,
    verbose=2,
    validation_data=(X_val, y_val),
)


plot_history(hist)


indices = np.random.choice(X_val.shape[0], 3)
preds = model.predict(X_val[indices])
plt.figure(figsize=(10, 10))
for i, idx in enumerate(indices):
    plt.subplot(3, 3, i + 1)
    plt.title("Image")
    plt.imshow(X_val[idx])
    plt.axis("off")
    plt.subplot(3, 3, i + 4)
    plt.title("Ground truth mask")
    plot_mask(y_val[idx])
    plt.axis("off")
    plt.subplot(3, 3, i + 7)
    plt.title("Predicted mask")
    plot_mask(preds[i])
    plt.axis("off")


def augment_image_and_mask(image, mask):
    """
    Apply data augmentation: horizontal flip, brightness change, and noise.
    Args:
      image: Input image tensor.
      mask: Corresponding mask tensor.
    Returns:
      Augmented image and mask.
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    mask = tf.cast(mask, tf.int32)

    flip = tf.random.uniform([]) > 0.5
    image = tf.image.flip_left_right(image)
    mask = tf.image.flip_left_right(mask)

    image = tf.image.random_brightness(image, max_delta=0.2)

    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, mask


indices_aug = np.random.choice(len(y_train), 50)
X_aug, y_aug = augment_image_and_mask(
    X_train[indices_aug],
    tf.expand_dims(tf.cast(y_train[indices_aug], tf.int32), axis=-1),
)
X_train_aug = np.concatenate([X_train, X_aug])
y_train_aug = np.concatenate([y_train, tf.squeeze(y_aug)])


plt.figure(figsize=(10, 12))
for i, idx in enumerate(indices_aug[:3]):
    plt.subplot(4, 3, i + 1)
    plt.imshow(X_train[idx])
    plt.title("Image before augmentaion")
    plt.axis("off")
    plt.subplot(4, 3, i + 4)
    plt.imshow(X_aug[i])
    plt.title("Image after augmentaion")
    plt.axis("off")
    plt.subplot(4, 3, i + 7)
    plt.imshow(y_train[idx])
    plt.title("Mask before augmentaion")
    plt.axis("off")
    plt.subplot(4, 3, i + 10)
    plt.imshow(y_aug[i])
    plt.title("Mask after augmentaion")
    plt.axis("off")
plt.show()


model = FastSCNN(CONFIG.num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(CONFIG.scheduler),
    loss=CONFIG.loss_function,
    metrics=["accuracy", dice_coefficient, iou_score],
)


hist = model.fit(
    X_train_aug,
    y_train_aug,
    batch_size=CONFIG.batch_size,
    epochs=100,
    verbose=2,
    validation_data=(X_val, y_val),
)


plot_history(hist)


indices = np.random.choice(X_val.shape[0], 3)
preds = model.predict(X_val[indices])
plt.figure(figsize=(10, 10))
for i, idx in enumerate(indices):
    plt.subplot(3, 3, i + 1)
    plt.title("Image")
    plt.imshow(X_val[idx])
    plt.axis("off")
    plt.subplot(3, 3, i + 4)
    plt.title("Ground truth mask")
    plot_mask(y_val[idx])
    plt.axis("off")
    plt.subplot(3, 3, i + 7)
    plt.title("Predicted mask")
    plot_mask(preds[i])
    plt.axis("off")


X_train_val, y_train_val = load_dataset(path, "camvid_trainval.txt")
X_test, y_test = load_dataset(path, "camvid_test.txt")


model = FastSCNN(CONFIG.num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(CONFIG.scheduler),
    loss=CONFIG.loss_function,
    metrics=["accuracy", dice_coefficient, iou_score],
)


hist = model.fit(
    X_train_val,
    y_train_val,
    batch_size=CONFIG.batch_size,
    epochs=100,
    verbose=2,
    validation_data=(X_test, y_test),
)


plot_history(hist)


indices = np.random.choice(X_val.shape[0], 10)
preds = model.predict(X_test[indices])
plt.figure(figsize=(10, 10))
for i, idx in enumerate(indices):
    row_offset = 0 if i < 5 else 10
    plt.subplot(6, 5, i + row_offset + 1)
    plt.title("Image")
    plt.imshow(X_test[idx])
    plt.axis("off")
    plt.subplot(6, 5, i + row_offset + 6)
    plt.title("Ground truth mask")
    plot_mask(y_test[idx])
    plt.axis("off")
    plt.subplot(6, 5, i + row_offset + 11)
    plt.title("Predicted mask")
    plot_mask(preds[i])
    plt.axis("off")
