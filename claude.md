# Project Rules and Guidelines - CA3 Medical Image Segmentation

This document outlines all rules, conventions, design decisions, and best practices used in the CA3 Medical Image Segmentation project.

## Table of Contents

1. [Code Style and Formatting](#code-style-and-formatting)
2. [Data Preprocessing Rules](#data-preprocessing-rules)
3. [Model Architecture Rules](#model-architecture-rules)
4. [Training Rules and Practices](#training-rules-and-practices)
5. [Loss Function Rules](#loss-function-rules)
6. [Evaluation Metrics Rules](#evaluation-metrics-rules)
7. [Visualization Rules](#visualization-rules)
8. [Reproducibility Rules](#reproducibility-rules)
9. [File Organization Rules](#file-organization-rules)
10. [Naming Conventions](#naming-conventions)

---

## Code Style and Formatting

### Python Code Style

- **Formatting**: Use Black-style formatting with 4 spaces for indentation
- **Line Length**: Maximum 88-100 characters per line
- **Imports**:
  - Group imports: standard library, third-party, local imports
  - Use absolute imports when possible
  - One import per line for clarity
- **Function Definitions**:
  - Use descriptive function names
  - Include docstrings for all functions
  - Use type hints where appropriate
- **Class Definitions**:
  - Use PascalCase for class names
  - Include comprehensive docstrings
  - Document all parameters and return values

### Code Example

```python
def get_slice_data(volume_data, seg_data, axis, slice_idx):
    """
    Extract 2D slice from 3D volume along specified axis.

    Args:
        volume_data: 3D numpy array of volume data
        seg_data: 3D numpy array of segmentation data
        axis: Axis along which to extract slice (0, 1, or 2)
        slice_idx: Index of slice to extract

    Returns:
        vol_slice_rotated: Rotated volume slice
        seg_slice_rotated: Rotated segmentation slice
    """
    # Implementation
```

### PyTorch Code Style

- Use `nn.Module` for all model architectures
- Use `torch.nn.functional` for stateless operations
- Always move models and data to the correct device
- Use `torch.no_grad()` for inference/evaluation
- Use `.eval()` and `.train()` to switch model modes

---

## Data Preprocessing Rules

### Slice Extraction Rules

1. **Axis Selection**: Extract slices from all three axes (0, 1, 2) for data diversity
2. **Starting Index**: Always start from index 10 to avoid edge artifacts
3. **Stride**: Use stride of 3 to ensure diversity while maintaining efficiency
4. **Maximum Slices**: Extract maximum of 48 slices per axis per volume
5. **Rotation**: Apply 90-degree rotation (k=1) to standardize orientation

### Padding Rules

1. **Target Dimensions**: Always pad to 256×256 pixels
2. **Padding Method**: Use zero-padding with constant value 0
3. **Padding Strategy**: Centered padding (equal padding on all sides when possible)
4. **Dimension Handling**: Ensure exact target dimensions by cropping if necessary

### Patch Extraction Rules

1. **Patch Size**: Extract 128×128 pixel patches
2. **Patch Locations**: Extract 4 non-overlapping patches per slice:
   - Top-left: (0, 0)
   - Top-right: (0, 128)
   - Bottom-left: (128, 0)
   - Bottom-right: (128, 128)
3. **Non-overlapping**: Patches must not overlap to maximize data diversity

### Segmentation Mask Rules

1. **Binarization**: Convert segmentation masks to binary (0 = background, 1 = foreground)
   - Rule: `seg_patch = (seg_patch > 0).astype(np.int64)`
   - This ensures masks are binary for 2-class segmentation
   - Prevents CUDA errors from multi-class labels
2. **Data Type**: Use `np.int64` for masks, `np.float32` for images
3. **Normalization**: Keep original intensity values (no normalization applied in preprocessing)

### Dataset Split Rules

1. **Split Ratio**: 70% train, 15% validation, 15% test
2. **Split Method**: Split at volume level (not patch level) to prevent data leakage
3. **Shuffling**: Shuffle volume indices before splitting
4. **Consistency**: Use same random seed for reproducible splits

---

## Model Architecture Rules

### U-Net Architecture Rules

1. **Encoder Structure**:
   - Use DoubleConv blocks (2× Conv2d + BatchNorm + ReLU)
   - Use MaxPool2d(2) for downsampling
   - Channel progression: 64 → 128 → 256 → 512 → 1024
2. **Decoder Structure**:
   - Use bilinear upsampling or transposed convolution
   - Concatenate with skip connections from encoder
   - Channel progression: 1024 → 512 → 256 → 128 → 64
3. **Skip Connections**:

   - Must connect encoder and decoder at same resolution levels
   - Use channel-wise concatenation (not addition)
   - Handle size mismatches with padding

4. **Output Layer**:
   - Use 1×1 convolution for final classification
   - Output channels = number of classes
   - No activation (raw logits for loss computation)

### Attention U-Net Rules

1. **Attention Gates**:
   - Place attention gates in all skip connections
   - Use gating signal from decoder, input from encoder
   - Apply sigmoid activation for attention coefficients
2. **Attention Computation**:
   - Transform both gating signal and input features
   - Use element-wise addition before attention
   - Upsample gating signal to match input spatial size

### General Architecture Rules

1. **Batch Normalization**: Use after every convolution (except output layer)
2. **Activation Functions**: Use ReLU for hidden layers, no activation for output
3. **Initialization**: Use default PyTorch initialization (Kaiming for ReLU)
4. **Bias Terms**: Set `bias=False` when using BatchNorm (redundant)

---

## Training Rules and Practices

### Hyperparameter Rules

1. **Learning Rate**:
   - Initial: 1e-4 (empirically determined)
   - Too high (>1e-3): Training instability
   - Too low (<1e-5): Slow convergence
2. **Batch Size**:
   - Default: 16
   - Balance between memory usage and gradient quality
   - Tested values: 8, 16, 32
3. **Weight Decay**:
   - Default: 1e-5
   - L2 regularization to prevent overfitting
   - Tested range: 1e-6 to 1e-4
4. **Epochs**:
   - Maximum: 50
   - Use early stopping to prevent overfitting
   - Typical convergence: 30-40 epochs

### Optimizer Rules

1. **Optimizer Choice**: Adam optimizer
   - Beta1: 0.9 (default)
   - Beta2: 0.999 (default)
   - Epsilon: 1e-8 (numerical stability)
2. **Learning Rate Scheduler**: ReduceLROnPlateau
   - Factor: 0.5 (halve learning rate)
   - Patience: 5 epochs
   - Mode: 'min' (monitor validation loss)
   - Verbose: True (print when reducing)

### Training Strategy Rules

1. **Early Stopping**:
   - Monitor: Validation Dice score
   - Patience: 10 epochs
   - Save best model based on validation performance
2. **Model Checkpointing**:
   - Save best model (highest validation Dice)
   - Save epoch number, model state, optimizer state
   - Include training history in checkpoint
3. **Gradient Clipping**:
   - Maximum gradient norm: 1.0
   - Prevents exploding gradients
   - Apply before optimizer step
4. **Mixed Precision**:
   - Use FP16 for forward pass (if GPU supports)
   - Use FP32 for loss computation
   - Improves speed and reduces memory

### Training Loop Rules

1. **Mode Switching**:
   - Always use `.train()` for training
   - Always use `.eval()` for validation/testing
   - Use `torch.no_grad()` for evaluation
2. **Gradient Management**:
   - Always call `optimizer.zero_grad()` before backward
   - Call `loss.backward()` to compute gradients
   - Call `optimizer.step()` to update parameters
3. **Metric Tracking**:
   - Track loss, Dice, IoU, accuracy every epoch
   - Track learning rate for analysis
   - Compute metrics on validation set every epoch

---

## Loss Function Rules

### Dice Loss Rules

1. **Smoothing Term**: Use epsilon = 1.0 to avoid division by zero
2. **Softmax Application**: Apply softmax to logits before Dice computation
3. **One-Hot Encoding**: Convert targets to one-hot for multi-class
4. **Averaging**: Average Dice across classes, then compute loss

### Cross-Entropy Loss Rules

1. **Input Format**: Use raw logits (no softmax)
2. **Target Format**: Use class indices (long tensor)
3. **Reduction**: Use default 'mean' reduction

### Combined Loss Rules

1. **Weight Balance**: Use equal weights (0.5, 0.5) for Dice and CE
2. **Rationale**:
   - Dice handles class imbalance
   - CE provides stable gradients
   - Combination ensures both global and local accuracy
3. **Computation Order**: Compute both losses, then combine

---

## Evaluation Metrics Rules

### Metric Computation Rules

1. **Dice Score**:
   - Compute per class, then average
   - Use same smoothing term (1.0) as in loss
   - Average across batch, then across samples
2. **IoU Score**:
   - Compute per class, then average
   - Use smoothing term for numerical stability
   - Average across batch, then across samples
3. **Pixel Accuracy**:
   - Compute as (correct pixels) / (total pixels)
   - Simple but can be misleading with class imbalance
4. **Per-Class Metrics**:
   - Always compute per-class Dice and IoU
   - Identify which classes are easier/harder to segment
   - Report both mean and per-class values

### Evaluation Rules

1. **Model State**: Always use `.eval()` mode
2. **Gradient Computation**: Use `torch.no_grad()` context
3. **Batch Processing**: Process test set in batches
4. **Metric Aggregation**: Aggregate across all test samples
5. **Statistical Analysis**: Compute mean, std, confidence intervals

---

## Visualization Rules

### Figure Generation Rules

1. **Resolution**: Save all figures at 300 DPI
2. **Format**: Use PNG format for quality
3. **Size**: Use appropriate figure sizes (typically 15×10 inches)
4. **Layout**: Use `plt.tight_layout()` before saving
5. **File Naming**: Use descriptive names with underscores

### Training Curves Rules

1. **Plot Both**: Always plot training and validation metrics together
2. **Line Styles**: Use different line styles/colors for train vs val
3. **Markers**: Use markers (o, s) for better visibility
4. **Grid**: Always include grid with alpha=0.3
5. **Labels**: Include axis labels, title, and legend
6. **Y-axis Limits**: Set appropriate limits (e.g., [0, 1] for metrics)

### Prediction Visualization Rules

1. **Layout**: Show input, ground truth, prediction side-by-side
2. **Colormaps**:
   - Use 'gray' for input images
   - Use 'jet' or 'viridis' for masks/predictions
   - Use consistent colormap ranges
3. **Overlays**: Show prediction overlays on input images
4. **Error Maps**: Visualize errors with color coding (red=FP, green=TN, etc.)
5. **Sample Selection**: Show diverse samples (best, worst, random)

### Statistical Visualization Rules

1. **Histograms**: Use appropriate bin counts (30-50 bins)
2. **Bar Charts**: Include value labels on bars
3. **Scatter Plots**: Include diagonal reference lines when appropriate
4. **Color Coding**: Use consistent color schemes
5. **Annotations**: Add mean/median lines with labels

---

## Reproducibility Rules

### Random Seed Rules

1. **Global Seed**: Use seed = 42 for all random operations
2. **Seed Everything Function**: Set seeds for:
   - Python random
   - NumPy random
   - PyTorch random (CPU and CUDA)
   - CUDA deterministic operations
3. **Seed Setting**: Call seed function at the beginning of script

### Deterministic Operations

1. **CUDA Deterministic**: Set `torch.backends.cudnn.deterministic = True`
2. **CUDA Benchmark**: Set `torch.backends.cudnn.benchmark = False`
3. **Python Hash Seed**: Set `os.environ['PYTHONHASHSEED'] = str(seed)`

### Data Loading Reproducibility

1. **Shuffle Seed**: Use same seed for DataLoader shuffling
2. **Worker Seed**: Set worker seed for multi-process data loading
3. **Dataset Order**: Ensure dataset order is deterministic

### Model Initialization

1. **Weight Initialization**: Use default PyTorch initialization (reproducible)
2. **Bias Initialization**: Use default initialization
3. **No Random Operations**: Avoid random operations in model forward pass

---

## File Organization Rules

### Directory Structure

```
CA3/
├── codes/
│   └── code.ipynb          # Main implementation notebook
├── dataset/
│   └── Q1_dataprep.py      # Data preprocessing module
├── description/
│   ├── NNDL_Assignment3.pdf
│   ├── Q1.pdf
│   └── Q2.pdf
├── report/
│   └── rep.tex              # LaTeX report
└── ref_files/
    └── NNDL_Assignment3.zip
```

### File Naming Conventions

1. **Python Files**: Use snake_case (e.g., `Q1_dataprep.py`)
2. **Notebook Files**: Use descriptive names (e.g., `code.ipynb`)
3. **Checkpoint Files**: Use descriptive names (e.g., `best_model.pth`)
4. **Figure Files**: Use descriptive names with underscores (e.g., `training_curves.png`)

### Code Organization Rules

1. **Imports First**: All imports at the top of file
2. **Constants**: Define configuration dictionaries at the top
3. **Functions**: Define helper functions before classes
4. **Classes**: Define model classes before training code
5. **Main Code**: Training/evaluation code at the end

---

## Naming Conventions

### Variable Naming

- **Snake Case**: Use for variables, functions, modules (e.g., `train_loader`, `calculate_dice_score`)
- **Pascal Case**: Use for classes (e.g., `UNet`, `IBSRPatchDataset`)
- **Constants**: Use UPPER_CASE (e.g., `CONFIG`, `BATCH_SIZE`)
- **Private**: Prefix with underscore for internal use (e.g., `_internal_method`)

### Function Naming

- **Verbs**: Use verb-based names (e.g., `get_slice_data`, `calculate_metrics`)
- **Descriptive**: Names should describe what function does
- **Consistent**: Use consistent naming patterns (e.g., `calculate_*` for metrics)

### Class Naming

- **Nouns**: Use noun-based names (e.g., `UNet`, `DiceLoss`)
- **Descriptive**: Names should describe the class purpose
- **Abbreviations**: Use standard abbreviations (e.g., `UNet`, not `U_Net`)

---

## Device Management Rules

### Device Selection Rules

1. **Priority Order**:
   - First: CUDA (if available)
   - Second: MPS (Apple Silicon)
   - Third: CPU
2. **Device Assignment**: Move model and data to same device
3. **Device Checking**: Always check device availability before use

### Memory Management Rules

1. **Batch Size**: Adjust based on available GPU memory
2. **Pin Memory**: Use `pin_memory=True` for CUDA (faster data transfer)
3. **Gradient Accumulation**: Use for larger effective batch sizes
4. **Mixed Precision**: Use to reduce memory usage

---

## Error Handling Rules

### Data Loading Errors

1. **Try-Except**: Wrap file loading in try-except blocks
2. **Warnings**: Use warnings for non-fatal errors
3. **Skip Files**: Skip problematic files with warning message
4. **Continue Processing**: Don't stop entire pipeline for single file error

### Training Errors

1. **Gradient Checking**: Check for NaN/Inf gradients
2. **Loss Checking**: Check for NaN/Inf losses
3. **Early Stopping**: Stop training if loss becomes NaN
4. **Checkpoint Recovery**: Save checkpoints regularly for recovery

---

## Documentation Rules

### Code Documentation

1. **Docstrings**: Include docstrings for all functions and classes
2. **Parameter Documentation**: Document all parameters with types and descriptions
3. **Return Documentation**: Document return values with types
4. **Example Usage**: Include usage examples in docstrings when helpful

### Report Documentation

1. **Mathematical Formulations**: Include equations for all loss functions and metrics
2. **Algorithm Descriptions**: Include pseudocode for complex algorithms
3. **Figure Captions**: Provide detailed captions explaining what figures show
4. **Table Captions**: Explain what tables contain and how to interpret

---

## Testing and Validation Rules

### Validation Rules

1. **Separate Sets**: Never use test set during training
2. **Validation Frequency**: Validate after every epoch
3. **Best Model**: Save model with best validation performance
4. **Early Stopping**: Use validation metrics for early stopping

### Test Set Rules

1. **Final Evaluation**: Only evaluate on test set once (after training)
2. **No Tuning**: Never tune hyperparameters based on test set
3. **Comprehensive Metrics**: Compute all metrics on test set
4. **Statistical Analysis**: Report mean, std, confidence intervals

---

## Configuration Management Rules

### Configuration Dictionary Rules

1. **Centralized**: Define all hyperparameters in CONFIG dictionary
2. **Descriptive Keys**: Use descriptive key names
3. **Comments**: Include comments explaining parameter choices
4. **Easy Modification**: Make it easy to change hyperparameters

### Example Configuration

```python
CONFIG = {
    'data_dir': '../dataset',
    'batch_size': 16,              # Balance memory and gradient quality
    'epochs': 50,                  # Maximum epochs (early stopping may stop earlier)
    'lr': 1e-4,                    # Empirically determined for stability
    'weight_decay': 1e-5,          # L2 regularization
    'num_workers': 2,               # Data loading workers
    'seed': 42,                     # Reproducibility seed
    'save_dir': './checkpoints',   # Model checkpoint directory
    'num_classes': 2,              # Background and foreground
    'patch_size': 128,             # Input patch size
    'in_channels': 1               # Grayscale images
}
```

---

## Best Practices Summary

1. **Always use `.eval()` and `torch.no_grad()` for evaluation**
2. **Always call `optimizer.zero_grad()` before backward pass**
3. **Always save best model based on validation metrics**
4. **Always use fixed random seeds for reproducibility**
5. **Always normalize/standardize data appropriately**
6. **Always use appropriate loss functions for the task**
7. **Always compute multiple evaluation metrics**
8. **Always visualize results comprehensively**
9. **Always document code and decisions**
10. **Always validate on separate validation set**

---

## Common Pitfalls to Avoid

1. **Don't evaluate on test set during training**
2. **Don't forget to switch model to eval mode**
3. **Don't forget to zero gradients before backward**
4. **Don't use test set for hyperparameter tuning**
5. **Don't forget to move data to correct device**
6. **Don't use inconsistent random seeds**
7. **Don't forget to handle class imbalance**
8. **Don't forget to save checkpoints regularly**
9. **Don't use wrong data types (e.g., float for masks)**
10. **Don't forget to binarize multi-class masks for binary segmentation**

---

## Project-Specific Rules

### Segmentation Mask Binarization

- **Rule**: Always binarize segmentation masks: `seg_patch = (seg_patch > 0).astype(np.int64)`
- **Reason**: Ensures binary masks (0=background, 1=foreground) for 2-class segmentation
- **Location**: Applied in `__getitem__` method of `IBSRPatchDataset`
- **Critical**: Prevents CUDA errors from unexpected multi-class label values

### Patch Extraction Strategy

- **Rule**: Extract exactly 4 non-overlapping patches per slice
- **Positions**: (0,0), (0,128), (128,0), (128,128) from 256×256 padded slice
- **Reason**: Maximizes data augmentation while maintaining spatial coherence

### Three-Axis Slice Extraction

- **Rule**: Extract slices from all three axes (0, 1, 2)
- **Reason**: Maximizes data diversity and captures different anatomical views
- **Benefit**: Significantly increases dataset size without additional volumes

---

## Version Control Rules

### Git Commit Rules

1. **Descriptive Messages**: Use clear, descriptive commit messages
2. **Logical Commits**: Group related changes together
3. **No Large Files**: Don't commit large model files or datasets
4. **Ignore Patterns**: Use .gitignore for checkpoints, figures, etc.

### Code Review Rules

1. **Self-Review**: Review own code before committing
2. **Documentation**: Ensure code is well-documented
3. **Testing**: Test code before committing
4. **Style Consistency**: Follow project style guidelines

---

## Performance Optimization Rules

### Training Optimization

1. **Batch Size**: Use largest batch size that fits in memory
2. **Data Loading**: Use multiple workers for data loading
3. **Mixed Precision**: Use when GPU supports it
4. **Gradient Accumulation**: Use for larger effective batch sizes

### Inference Optimization

1. **Batch Processing**: Process multiple samples together
2. **No Gradients**: Always use `torch.no_grad()` for inference
3. **Model Eval**: Always use `.eval()` mode for inference
4. **Device Optimization**: Keep data on GPU when possible

---

## Security and Privacy Rules

### Data Handling

1. **No Hardcoding**: Never hardcode paths or sensitive information
2. **Path Management**: Use relative paths or environment variables
3. **Data Privacy**: Follow medical data privacy regulations
4. **Access Control**: Restrict access to sensitive datasets

---

## Maintenance Rules

### Code Maintenance

1. **Regular Updates**: Keep dependencies up to date
2. **Documentation**: Update documentation with code changes
3. **Testing**: Test after major changes
4. **Refactoring**: Refactor code for clarity and maintainability

### Model Maintenance

1. **Version Control**: Version model checkpoints
2. **Metadata**: Save metadata with models (hyperparameters, etc.)
3. **Backup**: Keep backups of important models
4. **Documentation**: Document model performance and characteristics

---

## Conclusion

These rules ensure:

- **Consistency**: Consistent code style and practices across the project
- **Reproducibility**: Results can be reproduced by others
- **Maintainability**: Code is easy to understand and modify
- **Quality**: High-quality implementations and results
- **Best Practices**: Following industry and research best practices

Follow these rules throughout the project to ensure high-quality, reproducible, and maintainable code.
