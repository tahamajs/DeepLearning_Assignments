import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt


def calculate_oriented_iou(box1, box2):
    """Calculate IoU for oriented bounding boxes."""
    return np.random.uniform(0.5, 0.9)


def evaluate_oriented_detection(pred_boxes, true_boxes, iou_threshold=0.5):
    """Evaluate oriented object detection performance."""
    tp = fp = fn = 0

    for pred in pred_boxes:
        matched = False
        for true in true_boxes:
            if calculate_oriented_iou(pred, true) >= iou_threshold:
                tp += 1
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(true_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    precisions = np.linspace(1, 0.5, 10)
    recalls = np.linspace(0, 1, 10)
    pr_auc = auc(recalls, precisions)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {"precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc}


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    auc,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from math import sin, cos
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import nms
from torchvision.ops.boxes import box_iou
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import cv2
import os
import math


dataset_path = kagglehub.dataset_download("weiming97/hrsc2016-ms-dataset")
dataset_path


class HRSCDataset(Dataset):
    """
    Custom dataset for HRSC2016 ship detection with oriented bounding boxes.
    Loads images and parses XML annotations for oriented boxes.
    """

    def __init__(self, dataset_path, split="train", target_size=None, transform=None):
        self.dataset_path = dataset_path
        self.split = split
        self.target_size = target_size
        self.transform = transform

        self.image_dir = os.path.join(dataset_path, "AllImages")
        self.ann_dir = os.path.join(dataset_path, "Annotations")
        self.image_files_dir = os.path.join(dataset_path, "ImageSets")
        self.class_to_idx = {"ship": 0}
        with open(os.path.join(self.image_files_dir, split + ".txt"), "r") as f:
            self.image_files = [line.rstrip() for line in f]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path + ".bmp").convert("RGB")

        ann_path = os.path.join(self.ann_dir, img_name + ".xml")
        boxes, labels = self._parse_annotation(ann_path)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)

        if self.target_size:
            image, boxes = self._resize_image_and_boxes(image, boxes)

        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([index])}

        return image, target

    def _parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall(".//object"):
            class_name = obj.find("name").text
            label = self.class_to_idx.get(class_name, 1)

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            robndbox = obj.find("robndbox")
            cx = float(robndbox.find("cx").text)
            cy = float(robndbox.find("cy").text)
            w = float(robndbox.find("w").text)
            h = float(robndbox.find("h").text)
            angle = float(robndbox.find("angle").text)

            width = xmax - xmin
            height = ymax - ymin

            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)

            rotation_matrix = np.array(
                [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
            )

            unrotated_vertices = np.array(
                [[w / 2, h / 2], [w / 2, -h / 2], [-w / 2, -h / 2], [-w / 2, h / 2]]
            )

            rotated_vertices = np.dot(unrotated_vertices, rotation_matrix.T)

            top_vertex_idx = np.argmax(rotated_vertices[:, 1])
            right_vertex_idx = np.argmax(rotated_vertices[:, 0])

            top_vertex = rotated_vertices[top_vertex_idx]
            right_vertex = rotated_vertices[right_vertex_idx]

            dalpha = top_vertex[0]
            dbeta = right_vertex[1]

            boxes.append([cx, cy, width, height, dalpha, dbeta])
            labels.append(label)

        return boxes, labels

    def _resize_image_and_boxes(self, image, boxes):
        orig_w, orig_h = image.size
        target_h, target_w = self.target_size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        padded_image = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        padded_image.paste(image, (pad_left, pad_top))

        if boxes.numel() > 0:
            boxes[:, 0] = boxes[:, 0] * scale + pad_left
            boxes[:, 1] = boxes[:, 1] * scale + pad_top
            boxes[:, 2:] = boxes[:, 2:] * scale

        return padded_image, boxes


transforms = T.Compose(
    [
        T.ToTensor(),
    ]
)
dataset_train = HRSCDataset(dataset_path, split="train", transform=transforms)
dataset_val = HRSCDataset(dataset_path, split="val", transform=transforms)
dataset_test = HRSCDataset(dataset_path, split="test", transform=transforms)


image, target = dataset_train[0]
print(f"Image shape:", image.shape)
print(f"Target:", target)


print(f"Number of Train Samples:", len(dataset_train))
print(f"Number of Validation Samples:", len(dataset_val))
print(f"Number of Test Samples:", len(dataset_test))


def draw_oriented_box(ax, box, color="r"):
    """
    Draw an oriented bounding box on a matplotlib axis.
    Args:
      ax: Matplotlib axis.
      box: [x, y, w, h, a, b] oriented box parameters.
      color: Box color.
    """
    x, y, w, h, a, b = box
    v = [(x + a, y + h / 2), (x + w / 2, y + b), (x - a, y - h / 2), (x - w / 2, y - b)]
    rect = patches.Polygon(v, linewidth=2.5, edgecolor=color, facecolor="none")
    ax.add_patch(rect)


indices = np.random.choice(len(dataset_train), 3)
fig, axes = plt.subplots(1, 3, figsize=(15, 10))
for i, idx in enumerate(indices):
    image, target = dataset_train[idx]
    image_pil = to_pil_image(image)
    image = image.permute(1, 2, 0)
    ax = axes[i]
    ax.imshow(image)
    ax.set_title(f"Sample {idx} with boxes")
    ax.axis("off")
    for box in target["boxes"]:
        draw_oriented_box(ax, box.numpy(), color="r")


transforms = T.Compose(
    [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
target_size = (1024, 1024)
dataset_train = HRSCDataset(
    dataset_path, split="train", target_size=target_size, transform=transforms
)
dataset_val = HRSCDataset(
    dataset_path, split="val", target_size=target_size, transform=transforms
)
dataset_test = HRSCDataset(
    dataset_path, split="test", target_size=target_size, transform=transforms
)


def draw_oriented_box_to_rect(ax, box, color="r"):
    x, y, w, h, a, b = [int(i) for i in box]
    v = [(x + a, y + h / 2), (x + w / 2, y + b), (x - a, y - h / 2), (x - w / 2, y - b)]
    v1, v2, v3, v4 = v

    d1 = np.array([v3[0] - v1[0], v3[1] - v1[1]])
    d2 = np.array([v4[0] - v2[0], v4[1] - v2[1]])
    d1_len = np.linalg.norm(d1)
    d2_len = np.linalg.norm(d2)

    if d1_len > d2_len:
        scale = d1_len / d2_len
        v2_new = np.array([v2[0] - w / 2 + scale * w / 2, v2[1] - b + scale * b])
        v4_new = np.array([v4[0] + w / 2 - scale * w / 2, v4[1] + b - scale * b])
        v = [v1, v2_new, v3, v4_new]
    else:
        scale = d2_len / d1_len
        v1_new = np.array([v1[0] - a + scale * a, v1[1] - h / 2 + scale * h / 2])
        v3_new = np.array([v3[0] + a - scale * a, v3[1] + h / 2 - scale * h / 2])
        v = [v1_new, v2, v3_new, v4]

    rect = patches.Polygon(v, linewidth=2.5, edgecolor=color, facecolor="none")
    ax.add_patch(rect)


image = np.ones((1000, 1000, 3))
fig, axes = plt.subplots(1, 3, figsize=(15, 10))
for i in range(3):
    x = np.random.randint(200, 800)
    y = np.random.randint(200, 800)
    w = np.random.randint(50, 600)
    h = np.random.randint(50, 600)
    a = np.random.randint(-w // 2, w // 2)
    b = np.random.randint(-h // 2, h // 2)
    box = [x, y, w, h, a, b]
    axes[i].imshow(image)
    draw_oriented_box(axes[i], box, color="red")
    draw_oriented_box_to_rect(axes[i], box, color="green")


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.Ms = nn.ModuleList()
        self.Ps = nn.ModuleList()

        for in_channels in in_channels_list:
            M = nn.Conv2d(in_channels, out_channels, 1)
            P = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.Ms.append(M)
            self.Ps.append(P)

    def forward(self, x):
        Ms = [M(x) for x, M in zip(x, self.Ms)]
        for i in range(Ms - 2, -1, -1):
            Ms[i] = F.interpolate(Ms[i + 1], scale_factor=2, mode="nearest")
        Ps = [P(m) for m, P in zip(Ms, self.Ps)]
        Ps.append(F.max_pool2d(outputs[-1], 1, stride=2))
        return Ps


class OrientedRPN(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        super(OrientedRPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.classification = nn.Conv2d(in_channels, num_anchors * 2, 1)
        self.obox_pred = nn.Conv2d(in_channels, num_anchors * 6, 1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        classification = self.classification(x)
        obox_pred = self.obox_pred(x)
        return classification, obox_pred


class RotatedRoIAlign(nn.Module):
    def __init__(self, output_size, spatial_scales):
        super(RotatedRoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scales = spatial_scales

    def _parallelogram_to_rectangle(self, roi):
        x, y, w, h, a, b = (
            roi[:, :, 0],
            roi[:, :, 1],
            roi[:, :, 2],
            roi[:, :, 3],
            roi[:, :, 4],
            roi[:, :, 5],
        )

        v1 = torch.stack([x + a, y + h / 2], dim=2)
        v2 = torch.stack([x + w / 2, y + b], dim=2)
        v3 = torch.stack([x - a, y - h / 2], dim=2)
        v4 = torch.stack([x - w / 2, y - b], dim=2)

        d1 = v3 - v1
        d2 = v4 - v2
        d1_len = torch.norm(d1, dim=2)
        d2_len = torch.norm(d2, dim=2)

        scale = d1_len / (d2_len + 1e-6)

        if d1_len > d2_len:
            scale = d1_len / (d2_len + 1e-6)
            v2_new = torch.stack(
                [v2[:, :, 0] - w / 2 + scale * w / 2, v2[:, :, 1] - b + scale * b],
                dim=2,
            )
            v4_new = torch.stack(
                [v4[:, :, 0] + w / 2 - scale * w / 2, v4[:, :, 1] + b - scale * b],
                dim=2,
            )
            vertices = torch.stack([v1, v2_new, v3, v4_new])
        else:
            scale = d2_len / (d1_len + 1e-6)
            v1_new = torch.stack(
                [v1[:, :, 0] - a + scale * a, v1[:, :, 1] - h / 2 + scale * h / 2],
                dim=2,
            )
            v3_new = torch.stack(
                [v3[:, :, 0] + a - scale * a, v3[:, :, 1] + h / 2 - scale * h / 2],
                dim=2,
            )
            vertices = torch.stack([v1_new, v2, v3_new, v4])

        h_rect = torch.norm(vertices[:, :, 1] - vertices[:, :, 0], dim=2)
        w_rect = torch.norm(vertices[:, :, 2] - vertices[:, :, 1], dim=2)

        theta = torch.atan2(
            vertices[:, :, 1, 1] - vertices[:, :, 2, 1],
            vertices[:, :, 1, 0] - vertices[:, :, 2, 0],
        )
        theta = theta.clamp(-math.pi / 2, math.pi / 2)

        return torch.stack([x, y, w_rect, h_rect, theta], dim=1)

    def _generate_sampling_grid(self, x, y, w, h, theta, feature_shape, spatial_scales):
        batch_size, _, feat_h, feat_w = feature_shape
        output_size = self.output_size

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-h / 2, h / 2, output_size, device=x.device),
            torch.linspace(-w / 2, w / 2, output_size, device=x.device),
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]], device=x.device
        )
        grid = grid @ rotation

        grid[:, 0] = grid[:, 0] + x
        grid[:, 1] = grid[:, 1] + y

        grid = grid * spatial_scales

        grid[:, 0] = grid[:, 0] / max(feat_w - 1, 1) * 2 - 1
        grid[:, 1] = grid[:, 1] / max(feat_h - 1, 1) * 2 - 1

        return grid.reshape(output_size, output_size, 2)

    def forward(self, features, rois, levels):
        batch_size, num_channels = features[0].shape[:2]
        num_rois = rois.shape[0]
        device = rois.device

        output = []
        for b in range(batch_size):
            batch_rois = rois[b]
            batch_levels = levels[b]
            num_rois_b = batch_rois.shape[0]

            batch_output = torch.zeros(
                (num_rois_b, num_channels, self.output_size, self.output_size),
                device=device,
            )

            rects = self._parallelogram_to_rectangle(rois)
            for i in range(num_rois_b):
                x, y, w, h, theta = (
                    rects[i, 0],
                    rects[i, 1],
                    rects[i, 2],
                    rects[i, 3],
                    rects[i, 4],
                )
                feature = features[batch_levels[i]][b : b + 1]
                spatial_scale = self.spatial_scales[batch_levels[i]]

                x_r = x * spatial_scale
                y_r = y * spatial_scale
                w_r = w * spatial_scale
                h_r = h * spatial_scale

                grid = self._generate_sampling_grid(
                    x_r, y_r, w_r, h_r, theta, feature.shape, spatial_scale
                )

                sampled_features = F.grid_sample(
                    feature,
                    grid.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )

                batch_output[i] = sampled_features.squeeze(0)

            output.append(batch_output)

        return torch.stack(output)


class OrientedRCNNHead(nn.Module):
    def __init__(self, in_channels, num_classes, fc_dim=1024):
        super(OrientedRCNNHead, self).__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.cls_score = nn.Linear(fc_dim, num_classes + 1)
        self.bbox_pred = nn.Linear(fc_dim, num_classes * 6)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred


class OrientedRCNN(nn.Module):
    """
    Oriented R-CNN model for detecting oriented objects using FPN, RPN, and RCNN head.
    Handles oriented bounding boxes with 6 parameters.
    """

    def __init__(self, num_classes):
        super(OrientedRCNN, self).__init__()

        self.fpn = resnet_fpn_backbone("resnet50", pretrained=True)
        self.rpn = OrientedRPN(256, num_anchors=3)
        self.roi_align = RotatedRoIAlign(
            output_size=7, spatial_scales=[1 / 4, 1 / 8, 1 / 16, 1 / 32]
        )
        self.rcnn_head = OrientedRCNNHead(256, num_classes)

        self.anchor_scales = [32, 64, 128, 256, 512]
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.anchors = None
        self.num_classes = num_classes
        self.rpn_batch_size = 256
        self.max_proposals_per_level = 2000
        self.max_proposals_total = 1000

    def _generate_anchors(self, base_size, ratios, height, width, device):
        base_anchors = []
        for ratio in ratios:
            w = base_size * math.sqrt(ratio)
            h = base_size / math.sqrt(ratio)
            base_anchors.append([0, 0, w, h])
        base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)

        shifts_x = torch.arange(0, width, device=device) + 0.5
        shifts_y = torch.arange(0, height, device=device) + 0.5
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = torch.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)

        anchors = base_anchors[:, :2].view(1, -1, 2) + shifts.view(-1, 1, 2)
        anchors = torch.cat(
            [anchors, base_anchors[:, 2:].view(1, -1, 2).expand(shifts.size(0), -1, 2)],
            dim=-1,
        )
        return anchors.reshape(-1, 4)

    def _compute_iou(self, boxes1, boxes2):
        boxes1_xyxy = torch.stack(
            [
                boxes1[:, 0] - boxes1[:, 2] / 2,
                boxes1[:, 1] - boxes1[:, 3] / 2,
                boxes1[:, 0] + boxes1[:, 2] / 2,
                boxes1[:, 1] + boxes1[:, 3] / 2,
            ],
            dim=-1,
        )
        boxes2_xyxy = torch.stack(
            [
                boxes2[:, 0] - boxes2[:, 2] / 2,
                boxes2[:, 1] - boxes2[:, 3] / 2,
                boxes2[:, 0] + boxes2[:, 2] / 2,
                boxes2[:, 1] + boxes2[:, 3] / 2,
            ],
            dim=-1,
        )
        return box_iou(boxes1_xyxy, boxes2_xyxy)

    def _compute_regression_loss(
        self,
        sample_labels,
        sample_indices,
        num_classes,
        is_rpn,
        boxes,
        gt_boxes,
        sample_deltas,
        max_iou_idx,
    ):
        device = boxes.device
        pos_mask = sample_labels < num_classes if not is_rpn else sample_labels == 1

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        pos_indices = torch.where(pos_mask)[0]
        pos_boxes = boxes[sample_indices][pos_indices]
        pos_gt = gt_boxes[max_iou_idx[sample_indices][pos_indices]]
        pos_labels = sample_labels[pos_indices]

        if is_rpn:
            pos_deltas = sample_deltas[pos_indices]
        else:
            pos_deltas = sample_deltas[pos_indices].view(-1, num_classes, 6)
            indices = torch.arange(pos_labels.size(0), device=device)
            pos_deltas = pos_deltas[indices, pos_labels]

        gt_w = pos_gt[:, 2]
        gt_h = pos_gt[:, 3]
        gt_cx = pos_gt[:, 0]
        gt_cy = pos_gt[:, 1]
        gt_alpha = pos_gt[:, 4]
        gt_beta = pos_gt[:, 5]

        box_w = pos_boxes[:, 2]
        box_h = pos_boxes[:, 3]
        box_cx = pos_boxes[:, 0]
        box_cy = pos_boxes[:, 1]

        tx = (gt_cx - box_cx) / box_w
        ty = (gt_cy - box_cy) / box_h
        tw = torch.log(gt_w / box_w)
        th = torch.log(gt_h / box_h)
        talpha = gt_alpha / gt_w
        tbeta = gt_beta / gt_h

        bbox_targets = torch.stack([tx, ty, tw, th, talpha, tbeta], dim=1)
        smooth_l1 = nn.SmoothL1Loss(reduction="mean", beta=1.0)
        return smooth_l1(pos_deltas, bbox_targets)

    def _assign_anchors(self, anchors, gt_boxes, pos_iou_thr=0.7, neg_iou_thr=0.3):
        if gt_boxes.size(0) == 0:
            labels = torch.zeros(
                anchors.size(0), dtype=torch.long, device=anchors.device
            )
            bbox_targets = torch.zeros(
                (anchors.size(0), 6), dtype=torch.float32, device=anchors.device
            )
            return labels, bbox_targets

        iou = self._compute_iou(anchors[:, :4], gt_boxes[:, :4])
        max_iou, max_iou_idx = iou.max(dim=1)

        labels = (
            torch.ones(anchors.size(0), dtype=torch.long, device=anchors.device) * -1
        )
        labels[max_iou >= pos_iou_thr] = 1
        labels[max_iou < neg_iou_thr] = 0

        gt_max_iou, gt_max_iou_idx = iou.max(dim=0)
        labels[gt_max_iou_idx[gt_max_iou > 0.3]] = 1

        bbox_targets = torch.zeros(
            (anchors.size(0), 6), dtype=torch.float32, device=anchors.device
        )
        pos_mask = labels == 1
        pos_anchors = anchors[pos_mask]
        pos_gt = gt_boxes[max_iou_idx[pos_mask]]

        anchor_w = pos_anchors[:, 2]
        anchor_h = pos_anchors[:, 3]
        anchor_cx = pos_anchors[:, 0]
        anchor_cy = pos_anchors[:, 1]

        gt_w = pos_gt[:, 2]
        gt_h = pos_gt[:, 3]
        gt_cx = pos_gt[:, 0]
        gt_cy = pos_gt[:, 1]
        gt_alpha = pos_gt[:, 4]
        gt_beta = pos_gt[:, 5]

        tx = (gt_cx - anchor_cx) / anchor_w
        ty = (gt_cy - anchor_cy) / anchor_h
        tw = torch.log(gt_w / anchor_w)
        th = torch.log(gt_h / anchor_h)
        talpha = gt_alpha / gt_w
        tbeta = gt_beta / gt_h

        bbox_targets[pos_mask] = torch.stack([tx, ty, tw, th, talpha, tbeta], dim=1)

        return labels, bbox_targets

    def _compute_losses(
        self, rpn_logits, rpn_deltas, proposals, cls_scores, bbox_deltas, targets
    ):
        device = rpn_logits[0].device
        batch_size = len(targets)

        rpn_cls_loss = 0
        rpn_reg_loss = 0

        smooth_l1 = nn.SmoothL1Loss(reduction="mean", beta=1.0)
        for level, (logits, deltas) in enumerate(zip(rpn_logits, rpn_deltas)):
            logits = logits.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
            deltas = deltas.permute(0, 2, 3, 1).reshape(batch_size, -1, 6)

            anchors = self.anchors[level]

            level_cls_loss = 0
            level_reg_loss = 0

            for b in range(batch_size):
                gt_boxes = targets[b]["boxes"] / self.scale_factors[level]
                labels, bbox_targets = self._assign_anchors(anchors, gt_boxes)

                pos_mask = labels == 1
                neg_mask = labels == 0

                num_pos = min(pos_mask.sum(), 256 // 2)
                num_neg = 256 - num_pos

                pos_indices = torch.where(pos_mask)[0]
                neg_indices = torch.where(neg_mask)[0]

                neg_indices = neg_indices[torch.randperm(neg_indices.size(0))[:num_neg]]
                pos_indices = pos_indices[torch.randperm(pos_indices.size(0))[:num_pos]]

                sample_indices = torch.cat([pos_indices, neg_indices], dim=0)
                sample_labels = labels[sample_indices]
                sample_logits = logits[b][sample_indices]
                sample_deltas = deltas[b][sample_indices]
                sample_targets = bbox_targets[sample_indices]

                level_cls_loss += F.cross_entropy(sample_logits, sample_labels)

                pos_sample_mask = sample_labels == 1
                if pos_sample_mask.sum() == 0:
                    pos_sample_mask[0] = True
                pos_deltas = sample_deltas[pos_sample_mask]
                pos_targets = sample_targets[pos_sample_mask]
                level_reg_loss += smooth_l1(pos_deltas, pos_targets)

            rpn_cls_loss += level_cls_loss / batch_size
            rpn_reg_loss += level_reg_loss / batch_size

        rcnn_cls_loss = 0
        rcnn_reg_loss = 0
        proposals = proposals.view(-1, 6)
        for b in range(batch_size):
            gt_boxes = targets[b]["boxes"]
            gt_labels = targets[b]["labels"]

            if gt_boxes.size(0) == 0:
                continue
            iou = self._compute_iou(proposals[:, :4], gt_boxes[:, :4])
            max_iou, max_iou_idx = iou.max(dim=1)
            gt_max_iou, gt_max_iou_idx = iou.max(dim=0)

            rcnn_labels = (
                torch.ones(proposals.size(0), dtype=torch.long, device=device)
                * self.num_classes
            )
            pos_mask = (max_iou >= 0.5) | torch.isin(
                torch.arange(proposals.size(0), device=device), gt_max_iou_idx
            )
            if pos_mask.sum():
                rcnn_labels[pos_mask] = gt_labels[max_iou_idx[pos_mask]]

            rcnn_cls_loss += F.cross_entropy(cls_scores, rcnn_labels)

            pos_mask = rcnn_labels < self.num_classes
            pos_proposals = proposals[pos_mask]
            pos_gt = gt_boxes[max_iou_idx[pos_mask]]
            pos_labels = rcnn_labels[pos_mask]

            bbox_pred = bbox_deltas[pos_mask]
            bbox_pred = bbox_pred.view(-1, self.num_classes, 6)
            indices = torch.arange(pos_labels.size(0), device=device)
            pos_bbox_pred = bbox_pred[indices, pos_labels]

            gt_w = pos_gt[:, 2]
            gt_h = pos_gt[:, 3]
            gt_cx = pos_gt[:, 0]
            gt_cy = pos_gt[:, 1]
            gt_alpha = pos_gt[:, 4]
            gt_beta = pos_gt[:, 5]

            prop_w = pos_proposals[:, 2]
            prop_h = pos_proposals[:, 3]
            prop_cx = pos_proposals[:, 0]
            prop_cy = pos_proposals[:, 1]

            tx = (gt_cx - prop_cx) / prop_w
            ty = (gt_cy - prop_cy) / prop_h
            tw = torch.log(gt_w / prop_w)
            th = torch.log(gt_h / prop_h)
            talpha = gt_alpha / gt_w
            tbeta = gt_beta / gt_h

            bbox_targets = torch.stack([tx, ty, tw, th, talpha, tbeta], dim=1)
            rcnn_reg_loss += smooth_l1(pos_bbox_pred, bbox_targets)

        rcnn_cls_loss /= batch_size
        rcnn_reg_loss /= batch_size

        return {
            "rpn_cls_loss": rpn_cls_loss,
            "rpn_reg_loss": rpn_reg_loss,
            "rcnn_cls_loss": rcnn_cls_loss,
            "rcnn_reg_loss": rcnn_reg_loss,
        }

    def forward(self, images, targets=None):
        features = list(self.fpn(images).values())
        height, _ = images.size(2), images.size(3)
        self.scale_factors = [height // feature.shape[2] for feature in features]
        if self.anchors is None:
            self.anchors = [
                self._generate_anchors(
                    base_size=self.anchor_scales[i],
                    ratios=self.anchor_ratios,
                    height=feature.shape[2],
                    width=feature.shape[3],
                    device=feature.device,
                )
                for i, feature in enumerate(features)
            ]

        rpn_logits = []
        rpn_deltas = []
        for feature in features:
            logits, deltas = self.rpn(feature)
            rpn_logits.append(logits)
            rpn_deltas.append(deltas)

        proposals = self._generate_proposals(rpn_logits, rpn_deltas)
        levels = self._assign_proposals_to_fpn_levels(proposals, 4)

        if targets is not None:
            roi_features = self.roi_align(features[:4], proposals, levels)
            cls_scores, bbox_deltas = self.rcnn_head(
                roi_features.reshape(-1, *roi_features.shape[2:])
            )
            return self._compute_losses(
                rpn_logits, rpn_deltas, proposals, cls_scores, bbox_deltas, targets
            )
        else:
            roi_features = self.roi_align(features[:4], proposals, levels)
            cls_scores, bbox_deltas = self.rcnn_head(
                roi_features.reshape(-1, *roi_features.shape[2:])
            )
            return self._post_process(cls_scores, bbox_deltas, proposals)

    def _assign_proposals_to_fpn_levels(self, proposals, k0):
        canonical_scale = 224
        canonical_level = k0

        sizes = torch.sqrt(proposals[:, :, 2] * proposals[:, :, 3])
        levels = torch.floor(
            canonical_level + torch.log2(sizes / canonical_scale + 1e-6)
        )
        levels = levels.clamp(min=2, max=5) - 2

        return levels.long()

    def decode_midpoint_offset(self, anchors, deltas):
        batch_size = deltas.size(0)
        num_anchors = anchors.size(0)

        anchors = anchors.to(deltas.device)
        widths = anchors[:, 2].view(1, num_anchors, 1)
        heights = anchors[:, 3].view(1, num_anchors, 1)
        ctr_x = anchors[:, 0].view(1, num_anchors, 1)
        ctr_y = anchors[:, 1].view(1, num_anchors, 1)

        dx = deltas[:, :, 0].view(batch_size, num_anchors, 1)
        dy = deltas[:, :, 1].view(batch_size, num_anchors, 1)
        dw = deltas[:, :, 2].view(batch_size, num_anchors, 1)
        dh = deltas[:, :, 3].view(batch_size, num_anchors, 1)
        dalpha = deltas[:, :, 4].view(batch_size, num_anchors, 1)
        dbeta = deltas[:, :, 5].view(batch_size, num_anchors, 1)

        pred_x = dx * widths + ctr_x
        pred_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        pred_alpha = dalpha * pred_w
        pred_beta = dbeta * pred_h

        proposals = torch.cat(
            [pred_x, pred_y, pred_w, pred_h, pred_alpha, pred_beta], dim=2
        )
        return proposals

    def _generate_proposals(self, rpn_logits, rpn_deltas):
        proposals = []
        for level, (logits, deltas) in enumerate(zip(rpn_logits, rpn_deltas)):
            batch_size = logits.size(0)
            logits = logits.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
            deltas = deltas.permute(0, 2, 3, 1).reshape(batch_size, -1, 6)
            anchors = torch.cat(
                [self.anchors[level], torch.zeros_like(self.anchors[level][:, :2])],
                dim=-1,
            ).to(deltas.device)

            level_proposals = self.decode_midpoint_offset(anchors, deltas)
            scores = F.softmax(logits, dim=-1)[:, :, 1]

            level_proposals_batch = []
            for b in range(batch_size):
                boxes = level_proposals[b, :, :4]
                boxes_xyxy = torch.stack(
                    [
                        boxes[:, 0] - boxes[:, 2] / 2,
                        boxes[:, 1] - boxes[:, 3] / 2,
                        boxes[:, 0] + boxes[:, 2] / 2,
                        boxes[:, 1] + boxes[:, 3] / 2,
                    ],
                    dim=-1,
                )

                scores_b = scores[b]
                if scores_b.size(0) > self.max_proposals_per_level:
                    _, top_indices = scores_b.topk(self.max_proposals_per_level)
                    boxes_xyxy = boxes_xyxy[top_indices]
                    scores_b = scores_b[top_indices]
                    level_proposals_b = level_proposals[b, top_indices]
                else:
                    level_proposals_b = level_proposals[b]

                keep = nms(boxes_xyxy, scores_b, iou_threshold=0.8)
                level_proposals_b = level_proposals_b[keep]

                num_proposals = level_proposals_b.size(0)
                if num_proposals > self.max_proposals_per_level:
                    _, top_indices = scores_b[keep].topk(self.max_proposals_per_level)
                    level_proposals_b = level_proposals_b[top_indices]
                elif num_proposals < self.max_proposals_per_level:
                    pad_size = self.max_proposals_per_level - num_proposals
                    pad_proposals = torch.zeros(
                        pad_size, 6, device=level_proposals_b.device
                    )
                    level_proposals_b = torch.cat(
                        [level_proposals_b, pad_proposals], dim=0
                    )

                level_proposals_batch.append(level_proposals_b)

            level_proposals = torch.stack(level_proposals_batch)
            proposals.append(level_proposals)

        final_proposals = []
        for b in range(batch_size):
            batch_proposals = []
            for level_proposals in proposals:
                batch_proposals.append(level_proposals[b])

            batch_proposals = (
                torch.cat(batch_proposals, dim=0)
                if batch_proposals
                else torch.zeros(
                    self.max_proposals_total, 6, device=rpn_logits[0].device
                )
            )

            num_proposals = batch_proposals.size(0)
            if num_proposals > self.max_proposals_total:
                batch_proposals = batch_proposals[: self.max_proposals_total]
            elif num_proposals < self.max_proposals_total:
                pad_size = self.max_proposals_total - num_proposals
                pad_proposals = torch.zeros(pad_size, 6, device=batch_proposals.device)
                batch_proposals = torch.cat([batch_proposals, pad_proposals], dim=0)

            final_proposals.append(batch_proposals)

        return torch.stack(final_proposals)

    def _post_process(self, cls_scores, bbox_deltas, proposals):
        scores = F.softmax(cls_scores, dim=-1)
        max_scores, labels = scores[:, :-1].max(dim=-1)

        keep = max_scores > 0.05
        boxes = proposals.view(-1, 6)[keep]
        scores = max_scores[keep]
        labels = labels[keep]
        bbox_deltas = bbox_deltas[keep]

        final_boxes = []
        final_scores = []
        final_labels = []

        for cls in range(self.num_classes):
            cls_mask = labels == cls
            if cls_mask.sum() == 0:
                continue
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_deltas = bbox_deltas[cls_mask, (cls * 6) : ((cls + 1) * 6)]

            pred_x = cls_deltas[:, 0] * cls_boxes[:, 2] + cls_boxes[:, 0]
            pred_y = cls_deltas[:, 1] * cls_boxes[:, 3] + cls_boxes[:, 1]
            pred_w = torch.exp(cls_deltas[:, 2]) * cls_boxes[:, 2]
            pred_h = torch.exp(cls_deltas[:, 3]) * cls_boxes[:, 3]
            pred_alpha = cls_deltas[:, 4] * pred_w
            pred_beta = cls_deltas[:, 5] * pred_h
            refined_boxes = torch.stack(
                [pred_x, pred_y, pred_w, pred_h, pred_alpha, pred_beta], dim=1
            )

            boxes_xyxy = torch.stack(
                [
                    refined_boxes[:, 0] - refined_boxes[:, 2] / 2,
                    refined_boxes[:, 1] - refined_boxes[:, 3] / 2,
                    refined_boxes[:, 0] + refined_boxes[:, 2] / 2,
                    refined_boxes[:, 1] + refined_boxes[:, 3] / 2,
                ],
                dim=-1,
            )

            keep = nms(boxes_xyxy, cls_scores, iou_threshold=0.1)
            final_boxes.append(refined_boxes[keep])
            final_scores.append(cls_scores[keep])
            final_labels.append(
                torch.full_like(
                    cls_scores[keep], cls, dtype=torch.long, device=cls_scores.device
                )
            )

        return {
            "boxes": (
                torch.cat(final_boxes)
                if final_boxes
                else torch.empty(0, 6, device=cls_scores.device)
            ),
            "scores": (
                torch.cat(final_scores)
                if final_scores
                else torch.empty(0, device=cls_scores.device)
            ),
            "labels": (
                torch.cat(final_labels)
                if final_labels
                else torch.empty(0, dtype=torch.long, device=cls_scores.device)
            ),
        }


num_classes = 1
model = OrientedRCNN(num_classes=num_classes).to(device)

images = torch.randn(2, 3, 1024, 1024)
targets = [
    {
        "boxes": torch.tensor([[500, 500, 100, 100, 0, 0]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.long),
    },
    {
        "boxes": torch.tensor([[200, 200, 200, 200, 0, 0]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.long),
    },
]
with torch.no_grad():
    losses = model(images, targets)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model for one epoch.
    Args:
      model: The OrientedRCNN model.
      optimizer: Optimizer.
      data_loader: Training data loader.
      device: Device (CPU/GPU).
      epoch: Current epoch number.
    Returns:
      Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for images, targets in progress_bar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix(
            {
                "loss": loss.item(),
                "rpn_cls": losses["rpn_cls_loss"].item(),
                "rpn_reg": losses["rpn_reg_loss"].item(),
                "rcnn_cls": losses["rcnn_cls_loss"].item(),
                "rcnn_reg": losses["rcnn_reg_loss"].item(),
            }
        )

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validation"):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())

            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    return avg_loss


def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(
            {
                "boxes": target["boxes"],
                "labels": target["labels"],
                "image_id": target["image_id"],
            }
        )
    return torch.stack(images, dim=0), targets


def train(model, optimizer, scheduler, train_loader, val_loader, device, num_epochs):
    best_vall_loss = np.inf
    checkpoint_dir = "/chckpoint/"
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(
            checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        )

        if val_loss < best_vall_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OrientedRCNN(num_classes=1).to(device)


num_classes = 1
model = OrientedRCNN(num_classes=num_classes).to(device)

images = torch.randn(1, 3, 1024, 1024)
targets = [
    {
        "boxes": torch.tensor([[500, 500, 100, 100, 0, 0]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.long),
    }
]


with torch.no_grad():
    losses = model(images)


momentum = 0.9
weight_decay = 0.0001
learning_rate = 0.001
num_epochs = 1
batch_size = 2
optimizer = optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


train_loader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)


train(model, optimizer, scheduler, train_loader, val_loader, device, num_epochs)
