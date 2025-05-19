# Membership Inference Attack (MIA) In n Federated Learning 

## Overview

This project investigates Membership Inference Attacks (MIA) on image classification models using the CIFAR-10 and SVHN datasets.  
It allows you to train a CNN, perform advanced MIA evaluation, and analyze model privacy risks through a variety of features and visualizations.

---

## How the Attack Works

1. A convolutional neural network (SimpleCNN) is trained on the selected dataset.
2. For both training (members) and test (non-members) samples, features are extracted from model outputs:  
   - Maximum confidence
   - Softmax entropy
   - Probability assigned to the correct class
   - Gap between top two predicted probabilities
3. An attack model (logistic regression) is trained to classify samples as "member" or "non-member" using these features.
4. Attack performance is evaluated using accuracy, ROC-AUC, precision-recall, confusion matrix, and per-class analysis.

---

## Features

- Choose dataset (CIFAR-10 or SVHN) and number of epochs interactively
- Attack model uses multiple features, not just max confidence
- ROC and Precision-Recall curves for attack model
- Confusion matrix visualization
- Per-class ROC and confidence histograms
- Learning curves for model training/validation accuracy and loss
- Automated CSV logging of results, including all metrics and settings
- Generalization gap measurement (train vs. test accuracy)
- No datasets or outputs committed to the repository

---

## Project Structure

```

project\_root/
src/
main.py
train\_target.py
data\_loader.py
outputs/
.gitignore
README.md

````

---

## Installation

```sh
pip install torch torchvision scikit-learn matplotlib
````

---

## Usage Instructions

1. Navigate to the `src` directory:

   ```sh
   cd src
   ```
2. Run the main script:

   ```sh
   python main.py
   ```
3. Answer the prompts for dataset (`cifar10` or `svhn`), epochs (20/50/100), and train/load.

Outputs and results will be saved in a timestamped folder in `outputs/` at the project root.

---

## Outputs

Each run creates a folder such as `outputs/cifar10_epoch50_YYYY-MM-DD_HH-MM-SS/` with:

* `roc_curve.png`
* `pr_curve.png`
* `confusion_matrix.png`
* `accuracy_curve.png`
* `loss_curve.png`
* `results.csv`
* `class_X_roc_curve.png` and `class_X_confidence_histogram.png` for per-class analysis

**Sample output files can be found here:** \[your link here]

---

## Experiment Description

* **Goal:** Evaluate the vulnerability of trained image classifiers to membership inference attacks.
* **Datasets:** CIFAR-10 and SVHN
* **Model:** SimpleCNN (2 conv + 2 FC layers)
* **Attack:** Logistic regression using 4 features per sample
* **Metrics:** Attack accuracy, ROC-AUC, average precision, confusion matrix, per-class breakdowns, model generalization gap

---

## Understanding Output Files

| File                               | Description                          |
| ---------------------------------- | ------------------------------------ |
| `roc_curve.png`                    | ROC for MIA attack model             |
| `pr_curve.png`                     | Precision-Recall for attack model    |
| `confusion_matrix.png`             | Attack predictions vs. reality       |
| `accuracy_curve.png`               | Train/Validation accuracy vs. epochs |
| `loss_curve.png`                   | Train/Validation loss vs. epochs     |
| `results.csv`                      | Summary metrics for the run          |
| `class_X_roc_curve.png`            | ROC for attack on class X            |
| `class_X_confidence_histogram.png` | Confidence distributions for class X |

---

## Extending the Project

* Add more datasets by extending `data_loader.py`
* Try different models by editing `train_target.py`
* Simulate federated learning by splitting training data
* Test privacy defenses (dropout, DP, regularization)

---

## Reference Links

* [Paper: Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/1610.05820)
* [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
* [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)
* [TorchVision Docs](https://pytorch.org/vision/stable/index.html)
* [Sample Outputs](#) <!-- replace with your link -->

---

## Contact

Open an issue or pull request for questions, suggestions, or improvements.

---

```

---

**You can copy and paste this directly into your `README.md` for GitHub.**  
Just update `[Sample Outputs](#)` if you have a hosted results folder.

Let me know if you want a badge, credits, or anything else added!
```
