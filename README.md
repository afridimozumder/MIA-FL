# Membership Inference Attack (MIA) In n Federated Learning 
This repository explores **Membership Inference Attacks (MIA)** on image classification models using the **CIFAR-10** and **SVHN** datasets.  
You can use this code to train models, analyze their privacy vulnerabilities, and visualize/model how successful an attacker could be in distinguishing between "member" (training data) and "non-member" (test data) samples.

---

## Table of Contents

- [Overview](#overview)
- [How the Attack Works](#how-the-attack-works)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage Instructions](#usage-instructions)
- [Outputs and Results](#outputs-and-results)
- [Experiment Explanation](#experiment-explanation)
- [Understanding the Output Files](#understanding-the-output-files)
- [Extending and Modifying the Project](#extending-and-modifying-the-project)
- [Links](#links)

---

## Overview

A **membership inference attack** is a privacy attack where an adversary tries to determine if a particular data sample was used in the training of a machine learning model.  
This project provides a reproducible, extensible pipeline for running and analyzing MIA on classic deep learning models with easy-to-read outputs and code.

---

## How the Attack Works

1. **Model Training:**  
   A convolutional neural network (CNN) is trained on either CIFAR-10 or SVHN, using standard supervised learning.

2. **Feature Extraction:**  
   For both training data (members) and test data (non-members), various statistics of the model’s output (softmax) are computed, including:
   - Maximum confidence
   - Entropy
   - Probability assigned to the correct class
   - Gap between top two predicted probabilities

3. **Attack Model:**  
   A logistic regression model is trained to distinguish "member" vs. "non-member" samples using the above features.

4. **Evaluation:**  
   The code reports attack success using accuracy, ROC-AUC, precision-recall, confusion matrix, and per-class analysis.

---

## Features

- **Supports CIFAR-10 and SVHN datasets (image classification)**
- **Interactive:** Choose dataset, number of epochs, train/load option at runtime
- **Advanced attack features:** Uses not just max confidence, but entropy, correct class probability, and top-2 gap
- **Automated, reproducible outputs:**
  - ROC and Precision-Recall (PR) curves for the attack model
  - Confusion matrix visualization
  - Per-class attack analysis (ROC curves, confidence histograms)
  - Train/validation loss and accuracy curves
  - CSV run summary (accuracy, AUC, settings, confusion matrix, etc.)
- **Generalization gap measurement** (train vs. test accuracy)
- **No large data files in the repository** (datasets are downloaded on the fly)

---

## Project Structure


```

your\_project/
src/
main.py
train\_target.py
data\_loader.py
outputs/
\[auto-created with results for each run]
.gitignore
README.md

````

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo/src
````

2. **Install dependencies:**

   ```sh
   pip install torch torchvision scikit-learn matplotlib
   ```

---

## Usage Instructions

Simply run the main script (from inside the `src` directory):

```sh
python main.py
```

You will be prompted to select:

* **Dataset:** `cifar10` or `svhn`
* **Epochs:** 20, 50, or 100
* **Train/Load:** Train a new model, or load a previously trained model

Example prompt/response sequence:

```
Which dataset do you want to use? (cifar10/svhn): cifar10
How many epochs do you want to train for? (20/50/100): 50
Do you want to train a new model or load an existing model? (train/load): train
Outputs will be saved to: ../outputs/cifar10_epoch50_2025-05-19_14-23-00
Using device: cpu
...
```

**All output files (plots, CSV) will be saved in a timestamped folder in `outputs/` at the root of your repository.**

---

## Outputs and Results

Each experiment creates a folder, for example:

```
outputs/cifar10_epoch50_2025-05-19_14-23-00/
```

with files such as:

* `roc_curve.png` – ROC curve for attack model
* `pr_curve.png` – Precision-Recall curve
* `confusion_matrix.png` – Attack model confusion matrix
* `accuracy_curve.png` and `loss_curve.png` – Model training curves
* `class_0_roc_curve.png`, `class_0_confidence_histogram.png` – Per-class analysis
* `results.csv` – All summary metrics and settings

---

## Experiment Explanation

### What is the Experiment?

You are simulating a **real-world privacy attack** where a malicious party tries to figure out if a certain data point was part of a model’s training set. This is important for understanding privacy risks in both standard and federated learning settings.

### Steps:

1. **Train a CNN** on a dataset.
2. **Extract model outputs** for both training and test data.
3. **Build an attack model** to classify whether a sample is a member or non-member.
4. **Measure attack effectiveness** with accuracy, ROC-AUC, precision-recall, confusion, per-class breakdowns, etc.
5. **Log all outputs** for future comparison.

---

## Understanding the Output Files

| Output File/Plot                   | What It Shows                              | How to Interpret                                        |
| ---------------------------------- | ------------------------------------------ | ------------------------------------------------------- |
| `roc_curve.png`                    | ROC for membership attack                  | Higher AUC means attack is more successful              |
| `pr_curve.png`                     | Precision-Recall for attack                | High AP = good attacker performance (esp. on imbalance) |
| `confusion_matrix.png`             | Attack predictions vs. reality             | Diagonal = correct predictions                          |
| `accuracy_curve.png`               | Training/Validation accuracy vs. epochs    | Overfitting = big gap between curves                    |
| `loss_curve.png`                   | Training/Validation loss vs. epochs        |                                                         |
| `results.csv`                      | All numeric metrics/settings for this run  | Use for comparing experiments, making tables            |
| `class_X_roc_curve.png`            | ROC for attack on class X only             | Some classes may be easier to attack than others        |
| `class_X_confidence_histogram.png` | Distribution of max confidence for class X | Separation = easier attack for that class               |

---

## Advanced Features Used

* **Multiple attack features:** Instead of just using model confidence, attack model uses four features per sample.
* **Per-class breakdowns:** Find out if some classes are more vulnerable than others.
* **Training/validation curve plotting:** Know if your model is overfitting (a key risk factor for MIA).
* **Generalization gap logging:** Bigger gap = more MIA risk.
* **All results automatically logged** to CSV for reproducibility.

---

## Example Results Table (from results.csv)

| dataset | epochs | train\_acc | val\_acc | test\_acc | generalization\_gap | attack\_accuracy | attack\_roc\_auc | attack\_avg\_precision | optimal\_threshold | confusion\_matrix           |
| ------- | ------ | ---------- | -------- | --------- | ------------------- | ---------------- | ---------------- | ---------------------- | ------------------ | --------------------------- |
| cifar10 | 50     | 0.98       | 0.88     | 0.87      | 0.11                | 0.80             | 0.89             | 0.88                   | 0.61               | \[\[1500, 500],\[400,1600]] |

---

## Extending and Modifying the Project

* Add new datasets (e.g., MNIST, FashionMNIST) by implementing similar loader functions.
* Try different models (e.g., ResNet, MLP) by replacing `SimpleCNN`.
* Simulate federated learning by dividing the training data among multiple “clients,” train local models, and aggregate weights.
* Experiment with privacy defenses (dropout, differential privacy, regularization) and see their effect on MIA.

---

## Links

* [Paper: Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/1610.05820)
* [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)
* [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
* [Official TorchVision Documentation](https://pytorch.org/vision/stable/index.html)
* **Results/Output Folder:** [see here](https://drive.google.com/drive/folders/18dZ6dTzLg7nqobTcdiBfnPn0mj7cQFsb?usp=drive_link) 

---

## Contact

Questions? Suggestions?
Open an issue or reach out to the repo owner!

---

