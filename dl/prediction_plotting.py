from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import torch


def plot_classification_report(report, ax):
    labels = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    metrics = ["precision", "recall", "f1-score"]

    data = np.array([[report[label][metric] for metric in metrics] for label in labels])

    cax = ax.matshow(data, cmap="coolwarm")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    plt.colorbar(cax, ax=ax)

    # Adding the text
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Classes")
    ax.set_title("Classification Report")


def plot_confusion_matrix(y_true, y_pred, ax):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")


def plot_roc_curve(y_true, y_pred_prob, ax):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")


def plot_precision_recall_curve(y_true, y_pred_prob, ax):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    average_precision = average_precision_score(y_true, y_pred_prob)
    ax.plot(
        recall,
        precision,
        color="b",
        lw=2,
        label="Precision-Recall curve (area = %0.2f)" % average_precision,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")


def generate_report(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_original_samples = []
    sample_predictions = defaultdict(list)
    with torch.no_grad():
        for images, labels, original_samples in test_loader:
            images, labels = images.to(device), labels.to(device)
            # print(original_samples)
            outputs = model(images)
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_original_samples.extend(original_samples)

            # Collect predictions for each original sample
            for i, original_sample in enumerate(original_samples):
                sample_predictions[original_sample].append(preds[i].item())
    # Apply majority voting for each original sample
    majority_voted_labels = {}
    for original_sample, predictions in sample_predictions.items():
        majority_vote = max(set(predictions), key=predictions.count)
        majority_voted_labels[original_sample] = majority_vote

    # map the majority-voted labels back to the original samples
    fake_real = {0: "FAKE", 1: "REAL"}
    original_samples = [
        f"{original_sample}: {fake_real[majority_vote]}"
        for original_sample, majority_vote in majority_voted_labels.items()
        if original_sample.endswith("original")
    ]
    converted_samples = [
        f"{original_sample}: {fake_real[majority_vote]}"
        for original_sample, majority_vote in majority_voted_labels.items()
        if not original_sample.endswith("original")
    ]

    max_length = max(len(original_samples), len(converted_samples))
    original_samples.extend([""] * (max_length - len(original_samples)))
    converted_samples.extend([""] * (max_length - len(converted_samples)))

    df = pd.DataFrame(
        {"Original Samples": original_samples, "Converted Samples": converted_samples}
    )
    print(df.to_string(index=False))

    report = classification_report(
        all_labels, all_preds, target_names=["FAKE", "REAL"], output_dict=True
    )

    # Create subplots for visualization
    _, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot classification report
    plot_classification_report(report, axes[0, 0])

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, axes[0, 1])

    # ROC Curve and AUC
    plot_roc_curve(all_labels, all_preds, axes[1, 0])

    # Precision-Recall Curve
    plot_precision_recall_curve(all_labels, all_preds, axes[1, 1])

    plt.tight_layout()
    plt.show()
