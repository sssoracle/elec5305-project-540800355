"""
ELEC5305 Project
SVM and Random Forest Classification

This script:
1. Reads data/features.csv
2. Trains SVM and Random Forest classifiers
3. Reports Accuracy, Precision, Recall, and F1-score
4. Saves confusion matrices

Outputs:
    results/classification/classification_metrics.csv
    results/classification/*_confusion_matrix.png
"""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# 1. Basic Settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FEATURE_FILE = BASE_DIR / "data" / "features.csv"
CLASSIFICATION_DIR = BASE_DIR / "results" / "classification"
OPEN_IMAGES = True

CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)


def open_image(image_file):
    if OPEN_IMAGES:
        os.startfile(str(image_file))


# ============================================================
# 2. Load Features
# ============================================================

def load_features():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"\nFeature file not found:\n{FEATURE_FILE}\n\n"
            "Please run feature_extraction.py first."
        )

    df = pd.read_csv(FEATURE_FILE)
    x = df.drop(columns=["label", "sample_id"])
    y = df["label"]
    return x, y


# ============================================================
# 3. Model Training and Evaluation
# ============================================================

def build_models():
    return {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", C=10, gamma="scale")),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }


def evaluate_model(model_name, model, x_train, x_test, y_train, y_test, labels):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")

    filename = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    output_file = CLASSIFICATION_DIR / filename
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    open_image(output_file)

    print(f"\n{model_name}")
    for key, value in metrics.items():
        if key != "model":
            print(f"  {key}: {value:.4f}")
    print(f"  confusion matrix saved: {output_file}")

    return metrics


# ============================================================
# 4. Plot Metric Comparison
# ============================================================

def plot_metric_comparison(metrics_df):
    plot_df = metrics_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model")
    plt.ylim(0, 1.05)
    plt.title("Classification Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.3)

    output_file = CLASSIFICATION_DIR / "classification_metrics.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    open_image(output_file)

    print(f"Metric comparison saved: {output_file}")


# ============================================================
# 5. Main Program
# ============================================================

def main():
    print("\n==============================================")
    print("ELEC5305 Classification")
    print("==============================================")
    print(f"Feature file: {FEATURE_FILE}")
    print("==============================================\n")

    x, y = load_features()
    labels = sorted(y.unique())

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    metrics = []
    for model_name, model in build_models().items():
        result = evaluate_model(model_name, model, x_train, x_test, y_train, y_test, labels)
        metrics.append(result)

    metrics_df = pd.DataFrame(metrics)
    metrics_file = CLASSIFICATION_DIR / "classification_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)

    print(f"\nMetrics saved: {metrics_file}")
    plot_metric_comparison(metrics_df)
    print("\nClassification completed successfully.")


if __name__ == "__main__":
    main()
