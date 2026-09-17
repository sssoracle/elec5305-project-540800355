"""
ELEC5305 Project
Anti-Noise Experiment Under Different SNR Levels

This script:
1. Trains SVM and Random Forest on clean features
2. Generates noisy test signals under different SNR values
3. Evaluates Accuracy, Precision, Recall, and F1-score
4. Saves SNR performance tables and plots

Outputs:
    results/classification/snr_experiment_metrics.csv
    results/classification/snr_accuracy.png
    results/classification/snr_f1.png
"""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import (
    SAMPLES_PER_CLASS,
    add_awgn,
    build_dataset,
    extract_features,
    generate_signal,
)


# ============================================================
# 1. Basic Settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CLASSIFICATION_DIR = BASE_DIR / "results" / "classification"
SNR_LEVELS = [40, 30, 20, 10, 5, 0]
NOISY_SAMPLES_PER_CLASS = 60
OPEN_IMAGES = True

CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)


def open_image(image_file):
    if OPEN_IMAGES:
        os.startfile(str(image_file))


# ============================================================
# 2. Model Definitions
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


# ============================================================
# 3. Noisy Dataset Generation
# ============================================================

def build_noisy_dataset(snr_db):
    labels = [
        "Normal",
        "Voltage Sag",
        "Voltage Swell",
        "Interruption",
        "Harmonics",
        "Transient",
    ]

    rows = []
    for label in labels:
        for _ in range(NOISY_SAMPLES_PER_CLASS):
            clean_signal = generate_signal(label)
            noisy_signal = add_awgn(clean_signal, snr_db)
            features = extract_features(noisy_signal)
            features["label"] = label
            rows.append(features)

    df = pd.DataFrame(rows)
    x = df.drop(columns=["label"])
    y = df["label"]
    return x, y


# ============================================================
# 4. SNR Evaluation
# ============================================================

def evaluate_snr(models, x_train, y_train):
    for model in models.values():
        model.fit(x_train, y_train)

    rows = []

    for snr_db in SNR_LEVELS:
        x_test, y_test = build_noisy_dataset(snr_db)

        for model_name, model in models.items():
            y_pred = model.predict(x_test)

            rows.append(
                {
                    "model": model_name,
                    "snr_db": snr_db,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision_macro": precision_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "recall_macro": recall_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "f1_macro": f1_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 5. Plot SNR Results
# ============================================================

def plot_snr_metric(results_df, metric, filename):
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=results_df,
        x="snr_db",
        y=metric,
        hue="model",
        marker="o",
    )
    plt.gca().invert_xaxis()
    plt.ylim(0, 1.05)
    plt.title(f"{metric.replace('_', ' ').title()} Under Different SNR Levels")
    plt.xlabel("SNR (dB)")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)

    output_file = CLASSIFICATION_DIR / filename
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    open_image(output_file)

    print(f"Saved: {output_file}")


# ============================================================
# 6. Main Program
# ============================================================

def main():
    print("\n==============================================")
    print("ELEC5305 SNR Anti-Noise Experiment")
    print("==============================================")
    print(f"Clean samples/class: {SAMPLES_PER_CLASS}")
    print(f"Noisy samples/class: {NOISY_SAMPLES_PER_CLASS}")
    print(f"SNR levels         : {SNR_LEVELS}")
    print("==============================================\n")

    train_df = build_dataset()
    x_train = train_df.drop(columns=["label", "sample_id"])
    y_train = train_df["label"]

    results_df = evaluate_snr(build_models(), x_train, y_train)

    output_file = CLASSIFICATION_DIR / "snr_experiment_metrics.csv"
    results_df.to_csv(output_file, index=False)
    print(results_df)
    print(f"\nSNR metrics saved: {output_file}")

    plot_snr_metric(results_df, "accuracy", "snr_accuracy.png")
    plot_snr_metric(results_df, "f1_macro", "snr_f1.png")

    print("\nSNR experiment completed successfully.")


if __name__ == "__main__":
    main()
