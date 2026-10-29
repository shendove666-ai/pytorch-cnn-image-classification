# A Reproducible Study of CNN Architectures and Data Augmentation for Image Classification

## 1. Project Overview

This repository presents a reproducible deep learning engineering study
built with **PyTorch** on the **CIFAR-10** benchmark dataset. Designed
as an educational and methodological investigation into supervised
computer vision workflows, the project implements modular data
pipelines, interpretable Convolutional Neural Networks (CNNs),
controlled ablation experiments, and rigorous multi-metric evaluation.

> **Scope Statement:** This project is an empirical engineering study on
> a public benchmark dataset designed to demonstrate reproducible
> machine learning workflow design, controlled experimentation, and
> quantitative error analysis. It does not claim to propose novel deep
> learning algorithms.

## 2. Research Questions

The experimental pipeline is structured around three core research
questions: - **RQ1 (Baseline Capability):** Can an interpretable
two-block Convolutional Neural Network effectively learn discriminative
spatial representations for the 10-class CIFAR-10 classification task? -
**RQ2 (Data Augmentation Generalization):** How do standard spatial data
augmentations (random padded cropping and random horizontal flipping)
affect validation generalization and the train-validation performance
gap? - **RQ3 (Architectural Capacity & Efficiency):** How does adding a
third convolutional block with Batch Normalization, Dropout, and spatial
downsampling impact classification accuracy, parameter efficiency, and
training cost?

## 3. Dataset and Data Processing

-   **Dataset:** CIFAR-10 (`torchvision.datasets.CIFAR10`), containing
    60,000 `32x32` RGB images across 10 balanced classes.
-   **Validation Partitioning:** The 50,000 official training images are
    deterministically partitioned into **45,000 training images (90%)**
    and **5,000 validation images (10%)** using a fixed random seed
    (`seed=42`).
-   **Test Isolation:** The 10,000 official test images are strictly
    isolated and evaluated only once per experiment using the
    best-performing validation checkpoint.
-   **Normalization:** Channel-wise standardization using training set
    statistics: `mean=(0.4914, 0.4822, 0.4465)`,
    `std=(0.2470, 0.2435, 0.2616)`.

## 4. CNN Architectures

  ---------------------------------------------------------------------------------------------------
  Architecture    Convolutional Stages       Regularization     Flatten Dim            Trainable
                                                                                       Parameters
  --------------- -------------------------- ------------------ ---------------------- --------------
  `BaselineCNN`   2 blocks (`3 -> 32 -> 64`) Weight Decay       `64 x 8 x 8 = 4096`    `1,070,794`
                                             (`5e-4`)                                  

  `ImprovedCNN`   3 blocks                   `BatchNorm2d` +    `128 x 4 x 4 = 2048`   `620,586`
                  (`3 -> 32 -> 64 -> 128`)   `Dropout(0.3)` +                          
                                             Weight Decay                              
  ---------------------------------------------------------------------------------------------------

Both architectures output raw 10-dimensional logits directly passed to
`nn.CrossEntropyLoss` without redundant pre-loss Softmax layers.

## 5. Training Strategy

-   **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`).
-   **Optimizer:** Stochastic Gradient Descent (`torch.optim.SGD`) with
    learning rate `0.01`, momentum `0.9`, and L2 weight decay `5e-4`.
-   **Model Selection:** After each epoch, validation accuracy is
    computed under `model.eval()` and `torch.no_grad()`. The checkpoint
    achieving the highest validation accuracy is saved to disk and
    reloaded for final test evaluation.
-   **Run used for the results below:**
    `python scripts/run_experiments.py --epochs 5`, on CPU with seed
    `42`. Results represent this specific 5-epoch run.

## 6. Experimental Design

  ---------------------------------------------------------------------------------------
  Experiment ID         Model             Training Augmentation         Fixed Controls
                        Architecture                                    
  --------------------- ----------------- ----------------------------- -----------------
  **Experiment A**      `BaselineCNN`     None                          `seed=42`, same
  (`exp_a_baseline`)                                                    45k/5k/10k split,
                                                                        SGD (`lr=0.01`)

  **Experiment B**      `BaselineCNN`     `RandomCrop(32, pad=4)` +     Identical
  (`exp_b_augmented`)                     `RandomHorizontalFlip(0.5)`   architecture and
                                                                        hyperparameters
                                                                        to Exp A

  **Experiment C**      `ImprovedCNN`     `RandomCrop(32, pad=4)` +     Identical data
  (`exp_c_improved`)                      `RandomHorizontalFlip(0.5)`   pipeline and
                                                                        optimizer
                                                                        settings to Exp B
  ---------------------------------------------------------------------------------------

## 7. Evaluation Metrics

Models are evaluated on the held-out test set using: 1. Overall Test
Accuracy and Cross-Entropy Loss. 2. Per-class **Precision**, **Recall**,
and **F1-score** (`scikit-learn`). 3. Annotated `10 x 10` Confusion
Matrix heatmaps. 4. Qualitative inspection of high-confidence correct
and misclassified test samples.

## 8. Results and Interpretation

The table below reports the results from the completed 5-epoch CPU run.
Values are taken from `outputs/experiments_comparison.csv`.

  ------------------------------------------------------------------------------------------------------------
  Experiment   Model                Params Augmentation     Epochs Best Val  Final Test     Macro F1   Runtime
                                                                    Acc (%)     Acc (%)                    (s)
  ------------ --------------- ----------- -------------- -------- -------- ----------- ------------ ---------
  **Exp A**    `BaselineCNN`     1,070,794 No                    5    73.72   **72.91**   **0.7263**     318.4

  **Exp B**    `BaselineCNN`     1,070,794 Yes                   5    69.00       70.38       0.6982     349.3

  **Exp C**    `ImprovedCNN`       620,586 Yes                   5    71.28       71.89       0.7094     424.0
  ------------------------------------------------------------------------------------------------------------

### 8.1 Main observations

-   **Best result in this run:** Experiment A achieved the highest test
    accuracy (72.91%) and macro F1-score (0.7263).
-   **Effect of augmentation in this setup:** Experiment B's test
    accuracy (70.38%) was 2.53 percentage points lower than Experiment
    A's. This is an observation from this particular run, not evidence
    that data augmentation is universally harmful.
-   **Parameter efficiency:** Experiment C used 620,586 trainable
    parameters, approximately 42.0% fewer than the baseline's 1,070,794
    parameters. Its test accuracy was 71.89%, 1.02 percentage points
    below Experiment A.
-   **Runtime:** The recorded training time was about 318.4 s for A,
    349.3 s for B, and 424.0 s for C on CPU. Runtime depends on hardware
    and system load.

### 8.2 Limitations

Each experiment was run for only five epochs with one fixed seed (`42`).
Therefore, these results are useful as a reproducible engineering
comparison, but they do not establish statistical significance or prove
that one architecture or augmentation strategy is generally superior.
More epochs and repeated runs with multiple seeds would be needed for
stronger conclusions.

### 8.3 Experiment C visualizations

**Training curves**

![Experiment C training
curves](reports/figures/exp_c_improved_curves.png)

**Confusion matrix**

![Experiment C confusion
matrix](reports/figures/exp_c_improved_confusion_matrix.png)

**Prediction examples**

![Experiment C prediction
examples](reports/figures/exp_c_improved_prediction_examples.png)

## 9. Reproducibility

-   **Seed Control:** Fixed seeds across `random`, `numpy`, and `torch`
    (`seed=42`).
-   **Metadata Logging:** Exact package versions, hardware device names,
    and YAML configurations are automatically exported to
    `outputs/run_metadata.json`.
-   **Experiment outputs:** CSV metrics, training histories, model
    checkpoints, logs, and plots are written under `outputs/` and
    `reports/figures/`.

## 10. Installation and Usage

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/check_env.py
pytest -v
python scripts/run_smoke_test.py --with-data
python scripts/run_experiments.py --epochs 5
```
