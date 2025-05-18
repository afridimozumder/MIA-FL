import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

criterion = nn.CrossEntropyLoss()

def train_model(trainloader, valloader, epochs=100, model_path='cifar10_simplecnn.pth', output_dir=None):
    """
    Trains the SimpleCNN model and tracks train/validation accuracy and loss.
    Saves learning curves to output_dir.
    """
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0)

    # Track loss and accuracy for plotting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, "
              f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")

    # Plot learning curves and save
    if output_dir is not None:
        plt.figure()
        plt.plot(range(1, epochs+1), train_accuracies, label="Train Accuracy")
        plt.plot(range(1, epochs+1), val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train/Validation Accuracy Curve")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
        plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train/Validation Loss Curve")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()

    return model, train_accuracies, val_accuracies

def load_model(model_path, device):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def compute_attack_features(model, dataloader, device):
    """
    Returns a dictionary of attack features and ground truth labels:
    - max_confidence: maximum softmax confidence
    - entropy: entropy of softmax output
    - correct_conf: confidence assigned to correct class
    - top2_gap: gap between top 2 softmax values
    - pred_label: predicted class index
    - true_label: true class index
    """
    model.eval()
    max_confidences, entropies, correct_confidences, top2_gaps = [], [], [], []
    pred_labels, true_labels = [], []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = softmax(outputs)
            max_conf, pred = probs.max(dim=1)
            sorted_probs, _ = probs.sort(dim=1, descending=True)
            entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=1)  # avoid log(0)
            # For correct class probability
            batch_indices = torch.arange(inputs.size(0)).to(device)
            correct_conf = probs[batch_indices, labels.to(device)]
            top2_gap = sorted_probs[:, 0] - sorted_probs[:, 1]

            max_confidences.extend(max_conf.cpu().numpy())
            entropies.extend(entropy.cpu().numpy())
            correct_confidences.extend(correct_conf.cpu().numpy())
            top2_gaps.extend(top2_gap.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    features = {
        "max_confidence": np.array(max_confidences),
        "entropy": np.array(entropies),
        "correct_confidence": np.array(correct_confidences),
        "top2_gap": np.array(top2_gaps),
        "pred_label": np.array(pred_labels),
        "true_label": np.array(true_labels)
    }
    return features

def evaluate_mia(model, trainloader, testloader, device, output_dir, dataset_name="dataset", epochs=100, train_acc=None, val_acc=None):
    """
    Enhanced MIA evaluation with:
    - PR curve, ROC curve, Confusion matrix
    - Per-class ROC
    - CSV summary output
    - Multiple attack features
    - Generalization gap
    """
    print("Computing MIA attack features...")
    # Compute features for train (members) and test (non-members)
    member_features = compute_attack_features(model, trainloader, device)
    non_member_features = compute_attack_features(model, testloader, device)
    n_members = len(member_features["max_confidence"])
    n_nonmembers = len(non_member_features["max_confidence"])

    # Labels for MIA: 1=member, 0=non-member
    mia_labels = np.concatenate([np.ones(n_members), np.zeros(n_nonmembers)])
    # Concatenate features for both member and non-member
    def concat_feat(key): return np.concatenate([member_features[key], non_member_features[key]])
    attack_X = np.stack([
        concat_feat("max_confidence"),
        concat_feat("entropy"),
        concat_feat("correct_confidence"),
        concat_feat("top2_gap"),
    ], axis=1)

    # We use all four features above for the attack model
    from sklearn.linear_model import LogisticRegression
    attack_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    attack_model.fit(attack_X, mia_labels)
    mia_preds = attack_model.predict(attack_X)
    mia_probas = attack_model.predict_proba(attack_X)[:, 1]

    # Attack metrics
    attack_accuracy = accuracy_score(mia_labels, mia_preds)
    fpr, tpr, thresholds = roc_curve(mia_labels, mia_probas)
    roc_auc_val = auc(fpr, tpr)

    # Find optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Confusion matrix
    cm = confusion_matrix(mia_labels, mia_preds)

    # Precision-Recall
    precision, recall, pr_thresholds = precision_recall_curve(mia_labels, mia_probas)
    avg_precision = average_precision_score(mia_labels, mia_probas)

    # Per-class ROC for members
    # For each class, compute ROC AUC for distinguishing members vs. non-members for that class
    per_class_auc = {}
    for class_idx in range(10):
        class_labels = np.concatenate([
            (member_features["true_label"] == class_idx).astype(int),
            (non_member_features["true_label"] == class_idx).astype(int)
        ])
        class_membership = mia_labels
        if class_labels.sum() == 0:
            continue
        fpr_c, tpr_c, _ = roc_curve(class_membership[class_labels == 1], mia_probas[class_labels == 1])
        per_class_auc[class_idx] = auc(fpr_c, tpr_c)

        # Plot ROC curve for this class
        plt.figure()
        plt.plot(fpr_c, tpr_c, label=f'Class {class_idx} ROC (AUC={per_class_auc[class_idx]:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Per-Class ROC for Class {class_idx}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'class_{class_idx}_roc_curve.png'))
        plt.close()

        # Plot histogram of confidence for members/non-members of this class
        plt.figure()
        plt.hist(member_features["max_confidence"][member_features["true_label"] == class_idx], bins=30, alpha=0.5, label='Members')
        plt.hist(non_member_features["max_confidence"][non_member_features["true_label"] == class_idx], bins=30, alpha=0.5, label='Non-Members')
        plt.xlabel('Max Confidence')
        plt.ylabel('Frequency')
        plt.title(f'Confidence Distribution for Class {class_idx}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'class_{class_idx}_confidence_histogram.png'))
        plt.close()

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC={roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for MIA')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP={avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for MIA')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()

    # Confusion Matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Attack Model')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-member', 'Member'])
    plt.yticks(tick_marks, ['Non-member', 'Member'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Generalization gap: train acc - test acc
    # Compute train/test acc for the classifier model (not the attack model)
    model.eval()
    def accuracy(loader):
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        return correct / total

    final_train_acc = accuracy(trainloader)
    final_test_acc = accuracy(testloader)
    generalization_gap = final_train_acc - final_test_acc

    print(f"\n=== MIA Attack Summary ===")
    print(f"Attack accuracy: {attack_accuracy:.4f}")
    print(f"Attack ROC-AUC: {roc_auc_val:.4f}")
    print(f"Attack Average Precision: {avg_precision:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Generalization gap (train acc - test acc): {generalization_gap:.4f}")
    print(f"Final train accuracy: {final_train_acc:.4f}, Final test accuracy: {final_test_acc:.4f}")

    # Save attack summary results to results.csv
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "epochs", "train_acc", "val_acc", "test_acc", "generalization_gap",
            "attack_accuracy", "attack_roc_auc", "attack_avg_precision", "optimal_threshold",
            "confusion_matrix"
        ])
        writer.writerow([
            dataset_name, epochs, train_acc[-1] if train_acc else "",
            val_acc[-1] if val_acc else "", final_test_acc, generalization_gap,
            attack_accuracy, roc_auc_val, avg_precision, optimal_threshold, str(cm.tolist())
        ])
    print(f"Attack results saved to {csv_path}")
