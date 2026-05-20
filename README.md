# EfficientNet Transfer Learning Experiments — PyTorch

Transfer learning experiments comparing EfficientNet-B0 and EfficientNet-B2 on a 
pizza/steak/sushi image classification task, with results tracked via TensorBoard.

## What This Is

A structured experiment comparing model architecture, dataset size, and training 
duration across 8 configurations. The goal was to practice transfer learning and 
systematic experiment tracking.

## Experiments

| # | Model | Data Size | Epochs | Test Acc |
|---|-------|-----------|--------|----------|
| 1 | EfficientNet-B0 | 10% | 5 | 87.2% |
| 2 | EfficientNet-B0 | 10% | 10 | 87.6% |
| 3 | EfficientNet-B0 | 20% | 5 | 91.9% |
| 4 | EfficientNet-B0 | 20% | 10 | 88.9% |
| 5 | EfficientNet-B2 | 10% | 5 | 90.8% |
| 6 | EfficientNet-B2 | 10% | 10 | 89.8% |
| 7 | EfficientNet-B2 | 20% | 5 | 93.2% |
| 8 | EfficientNet-B2 | 20% | 10 | 95.9% |

**Key findings:**
- EfficientNet-B2 consistently outperforms B0 at equivalent configurations
- More data has a larger impact than more epochs
- Best result: B2 + 20% data + 10 epochs → **95.9% test accuracy**

## What I Learned

- Transfer learning with frozen feature extractors (EfficientNet-B0 and B2)
- Systematic experiment design across model, data size, and epoch variables
- TensorBoard experiment tracking with custom `SummaryWriter` per run
- ImageNet normalization and input size differences between B0 (224px) and B2 (260px)
- Modular training pipeline reused across all 8 experiments

## Project Structure
├── modules/
│   ├── data_setup.py      # Dataset loading and transforms
│   ├── engine.py          # Training and evaluation loops
│   ├── model_builder.py   # Model definitions
│   ├── train.py           # Training script
│   └── utils.py           # save_model and helpers
├── compare_models.ipynb   # Main experiment notebook
├── predict.ipynb          # Inference on custom images
├── runs-5-3-2026.zip      # TensorBoard logs
└── .gitignore


## Setup

### 1. Clone the repo
```bash
git clone https://github.com/cj00se/efficientnet-transfer-learning.git
cd efficientnet-transfer-learning
```

### 2. Install dependencies
```bash
pip install torch torchvision torchinfo tensorboard matplotlib
```

### 3. Add your data
Expects this structure:

### 4. Run experiments
Open `compare_models.ipynb` and run all cells.

### 5. View TensorBoard logs
```bash
tensorboard --logdir=runs
```

## Guided By

[Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) by Daniel Bourke

## Hardware
Trained on an NVIDIA RTX 5070 Ti (CUDA).

# Model weights
models/

# Data
pizz_steak_sushi_10/
pizza_steak_sushi_20/
pizza_steak_sushi/
train/
test/

# Python cache
__pycache__/
*.pyc

# Jupyter checkpoints
.ipynb_checkpoints/
