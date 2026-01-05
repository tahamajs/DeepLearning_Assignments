Here is the complete, detailed translation of the homework assignment, reorganized from the fragmented text into a structured, readable format.

---

# University of Tehran

School of Electrical and Computer Engineering

Course: Deep Learning and Neural Networks

Assignment #3

- **Designers:** Mohammad Jafari and Mehdi Sabour
- **Submission Deadline:** Azar 16, 1404 (approx. December 7, 2025)

---

## Question 1: Improved Brain Tissue Segmentation (Total Points: 85 + 5 Bonus)

In this question, you will implement a specific architecture for segmenting brain tissues into three classes: White Matter (WM), Grey Matter (GM), and Cerebrospinal Fluid (CSF). You will work with the **IBSR** dataset (T1W MRI images).

### 1-1. Data Preparation

**1-1-1. Research and Explanation (15 Points)**

- Read the provided paper and research the IBSR dataset.
- The dataset contains 18 patients. However, the labels provided in the raw data do not directly match the 3 classes (WM, GM, CSF).
- **Task:** Explain the pre-processing steps required to map the raw labels to the 4 specific classes (Background, CSF, GM, WM).
- **Code:** Complete the `HW3_Q1_dataprep.py` script.
  - You do not need to load all patient volumes at once to avoid RAM issues.
  - Load the `.nii.gz` files using the `nibabel` library.
  - Extract 2D slices from the 3D volumes.
  - Perform **Normalization** on the images.
  - Save the processed data in a format suitable for loading during training (e.g., `.npy`).
  - **Note:** The dataset includes 18 folders. Images are `IBSR_{patient no.}_ana_strip.nii.gz` and segmentation labels are `IBSR_{patient no.}_segTRI_fill_ana.nii.gz`.

**1-1-2. Data Visualization (10 Points)**

- Visualize the data for **3 different patients** .
- For each patient, display sample 2D slices from all three anatomical planes:
  1. **Axial** (x-y plane)
  2. **Sagittal** (y-z plane)
  3. **Coronal** (x-z plane)
- **Color Coding:** Display the ground truth labels with the following colors:
  - **Green:** CSF
  - **Yellow:** GM
  - **Brown:** WM
- Refer to Figure 1 (in original doc) for the expected visualization style.

### 1-2. Model Architecture Implementation (30 Points)

- Implement the **M-Net** architecture (a variation of U-Net) based on the provided paper.
- **Input:** 2D slices extracted in the previous step.
- **Structure:**
  - Encoder-Decoder structure.
  - Skip connections between corresponding layers.
  - The model should output pixel-wise classification for the 4 classes.
- Report the total number of trainable parameters.

### 1-3. Training Configuration & Initialization

**1-3-1. Optimization and Metrics (15 Points)**

- **Hyperparameters:** Use the values in Table 1 (below).
- **Loss Function:** Categorical Cross-Entropy.
- **Optimizer:** SGD.
- **Metrics:** Implement the following metrics from scratch (do not use `sklearn` or similar libraries for calculation):
  - Dice Coefficient
  - Jaccard Index
  - Precision
  - Recall
- Explain how True Positives (TP) and False Positives (FP) are extracted for the multi-class segmentation task.

Table 1: Hyperparameters

| Parameter | Value |

| :--- | :--- |

| Batch Size | 1 |

| Train / Validation Split | 20% / 80% (Note: prompt implies using a smaller subset for training speed or specific split logic) |

| Optimizer | SGD |

| Learning Rate | 0.0001 |

| Loss Function | Categorical Cross-Entropy |

| Epochs | 10 |

**1-3-2. Weight Initialization (LeCun Method) - (5 Points Bonus)**

- If weights are initialized too small, the signal vanishes; if too large, it explodes.
- **Task:** Implement **LeCun Initialization** manually (do not use `nn.init.kaiming` etc.).
- Initialize the weights of the Convolutional layers (`conv2d`) using a normal distribution with:

  - Mean = 0
  - Standard Deviation ($std$) calculated as:

    $$
    std = \sqrt{\frac{1}{fan\_in}}
    $$

  - Where $fan\_in$ is the number of input units to the tensor:

    $$
    fan\_in = n \times K_w \times K_h
    $$

    (where $n$ is the number of input channels, and $K$ is the kernel size).

  - _Hint:_ You can use `nn.init._calculate_fan_in_and_fan_out` to get the fan_in value.

- Explain the concepts of **Vanishing Gradients** and **Exploding Gradients** and how this initialization helps.

### 1-4. Results and Evaluation (15 Points)

- Train the model.
- **Plot:**
  - Loss vs. Epochs (Train and Validation).
  - Dice Coefficient vs. Epochs (Train and Validation).
- **Analysis:** Analyze the convergence of the model. Why did you get these results?
- **Target Accuracy:** You must achieve at least the following Dice scores on the validation set:
  - **CSF:** > 73%
  - **GM:** > 70%
  - **WM:** > 55%
- **Visualization:** For **4 random samples** from the validation set:
  - Show the original MRI slice.
  - Show the Ground Truth mask.
  - Show the Model Prediction mask.
  - (Ensure colors match the scheme in section 1-1-2).

---

## Question 2: Vehicle Detection using Faster R-CNN (Total Points: 100)

In this question, you will implement a vehicle detection system using the **Faster R-CNN** architecture on the **LSVH** (Large Scale Street View House Numbers) dataset, which has been adapted for vehicle detection.

### 2-1. Theoretical Questions (15 Points)

Answer the following questions based on the provided Faster R-CNN paper and general object detection knowledge:

1. What is the difference between **MobileNet** and **VGG16** backbones? What are the advantages/disadvantages of each?
2. What is the difference between **Soft-NMS** and standard **NMS** (Non-Maximum Suppression)?
3. Explain **Context-aware RoI Pooling** .

### 2-2. Dataset Preparation

**2-2-1. Pre-processing (15 Points)**

- **Dataset:** LSVH (Link provided in Google Drive).
- **Class Mapping:** The dataset contains various classes (car, bus, van, etc.). You must merge all vehicle-related classes into a single class labeled **"vehicle"** .
- **"Don't Care" Regions:** There are regions labeled as "don't care" (blurred or unrecognizable vehicles).
  - You must handle these during pre-processing so they are ignored during training (or removed), ensuring they do not confuse the model.
- **Bounding Boxes:** Extract and format the bounding box coordinates for the "vehicle" class.

**2-2-2. Data Visualization (5 Points)**

- Visualize the pre-processed data to verify the correctness of bounding boxes and labels.
- Draw bounding boxes on a few sample images (include "vehicle" labels).

### 2-3. Model Implementation (30 Points)

- Implement the **Faster R-CNN** architecture.
- **Backbone:** You must use **MobileNetV2** as the feature extractor (backbone).
- Follow the architecture described in the original Faster R-CNN paper (with the MobileNet modification).
- Report the total number of parameters.

### 2-4. Training (20 Points)

- Train the model using the hyperparameters in Table 2.
- **Data Split:**
  - Use **40%** of the dataset for Training (to save computational time/resources).
  - From that 40%: Split into Train (90%) and Validation (10%) for internal evaluation.
  - Use another **20%** of the original data for Testing.
- **Metrics:**
  - Calculate **mAP@50** (Mean Average Precision at IoU threshold 0.5).
  - Report **Precision** and **Recall** .
- **Constraint:** The model must achieve **mAP@50 > 60%** on the validation set.

Table 2: Hyperparameters

| Parameter | Value |

| :--- | :--- |

| Batch Size | 8 |

| Optimizer | Adam |

| Learning Rate | Based on paper / standard practice |

| Loss Function | Original Fast R-CNN Loss (Classification + Regression) |

| Epochs | 50 |

### 2-5. Results and Evaluation (15 Points)

- **Plots:**
  - Plot the Total Loss per epoch.
  - Plot mAP@50 per epoch.
- **Visual Evaluation:**
  - Perform inference on **4 images** from the Test set.
  - Draw the predicted Bounding Boxes and the Confidence Scores on these images.
- **Analysis:** Discuss the model's stability, convergence, and any issues faced (e.g., did the loss decrease smoothly?).
