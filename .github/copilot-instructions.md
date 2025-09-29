# AI Coding Guidelines for DRL Course Repository

## Project Overview

This is an educational repository for Deep Reinforcement Learning (DRL) course assignments at Sharif University of Technology. Each Computer Assignment (CA) implements RL algorithms from basic concepts to advanced topics like quantum RL and neuromorphic computing.

## Architecture & Structure

- **Modular Design**: Each CA has `agents/`, `environments/`, `experiments/`, `models/`, `utils/` subdirectories
- **Progressive Complexity**: CAs build from fundamentals (CA1: MDP basics) to cutting-edge (CA19: quantum RL)
- **Framework Stack**: PyTorch for neural networks, Gymnasium for environments, NumPy/Matplotlib for computation/visualization

## Key Patterns & Conventions

### Agent Implementations

- **Class Naming**: `{Algorithm}Agent` (e.g., `DQNAgent`, `REINFORCEAgent`)
- **Core Methods**: `select_action()`, `update()`, `train()`
- **Training Functions**: `train_{algorithm}_agent()` returning score history
- **Example**: `agents/dqn_agent.py` in CA4 shows DQN with experience replay and target networks

### Neural Network Models

- **Base Class**: Inherit from `torch.nn.Module`
- **Naming**: `{Algorithm}Network` or `{Component}Network` (e.g., `PolicyNetwork`, `ValueNetwork`)
- **Forward Pass**: Return action distributions or Q-values
- **Example**: `models/policy_network.py` uses MLP with softmax for discrete policies

### Training Loops

- **Progress Tracking**: Use `tqdm` for episode progress bars
- **Metrics**: Track episode rewards, losses, and convergence metrics
- **Evaluation**: Run evaluation episodes every N training episodes
- **Example**: Training functions in `experiments/` return dict with `'scores'`, `'losses'`, `'eval_scores'`

### Environment Wrappers

- **Standardization**: Wrap Gymnasium envs for consistent interfaces
- **Preprocessing**: Normalize states, handle action spaces
- **Example**: `environments/wrappers.py` provides `NormalizedEnv` and `FrameStack`

## Critical Workflows

### Running Experiments

```bash
# Install dependencies (run once)
pip install -r requirements.txt

# Train agent (example from CA4)
python -m experiments.train_dqn --env CartPole-v1 --episodes 1000

# Run notebook for interactive exploration
jupyter notebook CA4.ipynb
```

### Debugging Training

- **Loss Monitoring**: Plot policy/value losses over time
- **Reward Curves**: Use moving averages to detect convergence
- **Gradient Checks**: Verify gradients flow through networks
- **Environment Reset**: Ensure proper env resets between episodes

### Hyperparameter Tuning

- **Grid Search**: Use `experiments/hyperparameter_sweep.py` patterns
- **Logging**: Save configs and results to JSON/CSV
- **Reproducibility**: Set seeds with `utils.set_seed(seed)`

## Integration Points

### External Dependencies

- **PyTorch**: Use `torch.nn.functional` for activations, `torch.optim` for optimizers
- **Gymnasium**: Register custom envs, handle both discrete/continuous spaces
- **Visualization**: Seaborn for statistical plots, Matplotlib for curves

### Cross-CA Communication

- **Shared Utils**: Common functions in `utils/` (e.g., `plot_learning_curve()`)
- **Environment Interfaces**: Consistent `reset()`, `step()` across CAs
- **Model Loading**: Save/load PyTorch models with `torch.save/load`

## Common Pitfalls & Solutions

### Training Instability

- **Gradient Clipping**: Clip gradients at 1.0 in policy gradients
- **Target Networks**: Use in DQN/Actor-Critic for stability
- **Entropy Bonus**: Add to continuous action policies for exploration

### Sample Efficiency

- **Experience Replay**: Buffer size 10k-100k for DQN variants
- **Prioritized Replay**: Use for important transitions
- **Model-Based**: Implement Dyna-Q for planning (see CA10)

### Environment Handling

- **Action Spaces**: Check discrete vs continuous, handle properly
- **Observation Spaces**: Flatten/box handling for neural nets
- **Termination**: Respect `done` flags, don't update after episode end

## File Organization Examples

- `CA4/agents/dqn_agent.py`: DQN implementation with Double/Dueling variants
- `CA6/experiments/policy_gradient_comparison.py`: Compare REINFORCE vs Actor-Critic
- `CA10/models/neural_dynamics.py`: Neural network for environment modeling

Focus on implementing RL algorithms correctly, following PyTorch best practices, and ensuring reproducible results through proper seeding and logging.



## Table of Contents

1. [Document Structure Overview](#document-structure-overview)
2. [Header and Personal Information](#header-and-personal-information)
3. [Table of Contents](#table-of-contents)
4. [Executive Summary](#executive-summary)
5. [Objectives](#objectives)
6. [Evaluation Plan &amp; Metrics](#evaluation-plan--metrics)
7. [Reproducibility &amp; Environment](#reproducibility--environment)
8. [Data Preparation and Preprocessing](#data-preparation-and-preprocessing)
9. [Model Architecture](#model-architecture)
10. [Training and Optimization](#training-and-optimization)
11. [Results and Analysis](#results-and-analysis)
12. [Visualization Guidelines](#visualization-guidelines)
13. [Conclusion](#conclusion)
14. [Code Style and Documentation](#code-style-and-documentation)
15. [References and Citations](#references-and-citations)

## Document Structure Overview

This style guide provides standardized formatting and content guidelines for all Neural Networks and Deep Learning assignment notebooks. Each notebook must follow IEEE-style documentation with clear, concise explanations limited to essential information. The structure ensures professional presentation while maintaining readability for academic and research audiences.

## Header and Personal Information

**Format:** HTML table with university logos and centered personal information.

**Required Elements:**

- University logos (left and right alignment)
- Course title: "Neural Networks and Deep Learning"
- Student name and ID
- University and faculty information

**Example:**

```html
<div
  style="display:block;width:100%;margin:auto;"
  direction="rtl"
  align="center"
>
  <table style="border-style:hidden;border-collapse:collapse;">
    <tr>
      <td><img width="130" align="right" src="logo1.png" /></td>
      <td style="text-align:center;">
        <h1>Neural Networks and Deep Learning</h1>
        <h2>Student Name - Student ID</h2>
        <h3>University Name<br />Faculty Name</h3>
      </td>
      <td><img width="170" align="left" src="logo2.png" /></td>
    </tr>
  </table>
</div>
```

## Table of Contents

**Format:** Automatically generated using VS Code TOC extension.

**Requirements:**

- Hierarchical structure with proper nesting
- Links to all major sections
- Consistent numbering and indentation
- Include subsections for complex analyses

**TOC Configuration:**

```
numbering=false
anchor=true
flat=false
minLevel=1
maxLevel=6
```

## Executive Summary

**Length:** 4-6 lines in IEEE format.

**Structure:**

1. Brief problem statement and methodology overview
2. Key quantitative results (accuracy, metrics)
3. Main findings and significance
4. Comparison with baselines (if applicable)

**Example:**
This notebook implements [method] for [task], achieving [key metric: value] with [optimization technique]. The analysis demonstrates [main finding], with comprehensive evaluation showing [comparison results]. Key contributions include [2-3 bullet points of results].

## Objectives

**Format:** Bullet point list (4-6 items).

**Requirements:**

- Specific, measurable objectives
- Clear technical goals
- Evaluation methodology
- Comparison criteria

**Example:**

- Implement [specific model/architecture] for [task]
- Optimize [hyperparameters] using [method]
- Evaluate using [specific metrics] appropriate for [task type]
- Compare with [baseline methods]

## Evaluation Plan & Metrics

**Length:** 6-8 lines explaining metric selection rationale.

**Structure:**

1. Task-specific evaluation requirements
2. Primary and secondary metrics
3. Rationale for metric choice
4. Helper function descriptions

**Example:**
Given the [task characteristics], models are evaluated using [primary metrics] for [reason]. Secondary metrics include [additional metrics] to assess [specific aspects]. [Brief explanation of why these metrics are appropriate].

## Reproducibility & Environment

**Format:** Dedicated section with environment details.

**Required Elements:**

- Python version and key libraries with versions
- Hardware specifications (if relevant)
- Random seed settings
- Data source information

**Example:**

- Python 3.8+, PyTorch 1.9.0, scikit-learn 1.0.2
- CUDA 11.1, GPU: RTX 3080
- Random seed: 42 for reproducibility
- Dataset: [source and version]

## Data Preparation and Preprocessing

**Structure:**

1. Data loading and initial exploration
2. Preprocessing steps with justification
3. Train/validation/test splits
4. Data augmentation (if applicable)

**Documentation Requirements:**

- Dataset statistics and characteristics
- Preprocessing rationale
- Split ratios and stratification

## Model Architecture

**Format:** Clear architectural description with diagrams.

**Requirements:**

- Layer-by-layer specification
- Parameter counts
- Architectural choices rationale
- Input/output dimensions

**Example:**
The proposed architecture consists of [overview]. Key components include [main layers], with [justification for design choices]. Total parameters: [count], designed for [specific requirements].

## Training and Optimization

**Structure:**

1. Training configuration
2. Optimization algorithm and hyperparameters
3. Learning rate scheduling
4. Regularization techniques

**Documentation:**

- Batch size, epochs, early stopping criteria
- Loss function selection rationale
- Convergence analysis

## Results and Analysis

**Format:** Quantitative results followed by qualitative analysis.

**Structure:**

1. Performance metrics table/comparison
2. Key findings (3-5 lines per major result)
3. Comparative analysis
4. Ablation studies (if applicable)

**Visualization Requirements:**

- All plots saved to `visualization/` folder
- Clear labeling and legends
- Consistent color schemes
- Error bars for statistical significance

## Visualization Guidelines

**Requirements:**

- All output images moved to `visualization/` folder
- Consistent naming: `notebook_section_plot_type.png`
- High-resolution exports (300 DPI)
- Professional color schemes
- Clear axis labels and legends

**Example Structure:**

```
visualization/
├── CA1_fraud_detection_roc_curve.png
├── CA1_confusion_matrix_comparison.png
├── CA2_covid_accuracy_comparison.png
└── ...
```

## Conclusion

**Length:** 8-10 lines summarizing entire work.

**Structure:**

1. Summary of achieved objectives
2. Key technical contributions
3. Performance highlights
4. Limitations and future work
5. Broader implications

**Example:**
This work successfully [main achievement], demonstrating [key finding]. The implemented [method] achieved [quantitative results], outperforming [baselines] by [margin]. Key contributions include [3-4 bullet points]. Future work could explore [extensions].

## Code Style and Documentation

**Requirements:**

- Clear, commented code with docstrings
- Consistent naming conventions
- Modular function design
- Error handling and validation

**Documentation Standards:**

- Function docstrings with parameters and returns
- Inline comments for complex logic
- Markdown explanations before code blocks
- Reproducible random seeds

## References and Citations

**Format:** IEEE citation style.

**Requirements:**

- All external libraries and papers cited
- Dataset sources referenced
- Implementation references included

**Example:**
[1] Authors, "Paper Title," Journal, vol. X, no. Y, pp. Z, Year.
[2] Library Name, Version, "URL", accessed Date.

---

## Implementation Checklist

- [ ] Header with personal information and logos
- [ ] Table of contents with proper links
- [ ] Executive summary (4-6 lines)
- [ ] Clear objectives (4-6 bullets)
- [ ] Evaluation plan with rationale
- [ ] Reproducibility information
- [ ] Well-documented code sections
- [ ] Results with 3-5 line explanations
- [ ] Professional visualizations in dedicated folder
- [ ] Conclusion (8-10 lines)
- [ ] Proper citations and references

## Quality Assurance

- All explanations limited to specified line counts
- IEEE-style professional writing
- Consistent formatting across all notebooks
- All images moved to visualization folder
- Code is well-commented and reproducible
