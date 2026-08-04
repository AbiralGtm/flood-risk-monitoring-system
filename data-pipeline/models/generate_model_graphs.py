"""
generate_model_graphs.py

PURPOSE
-------
Generates the standard set of evaluation visuals for your trained flood
risk classification model, suitable for your report's Testing &
Evaluation section (matches the metrics your proposal already commits
to: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix).

INPUT
------
model_xgboost.joblib          (or model_random_forest.joblib)
ml_training_dataset_v2.csv    (the same file used for training)

OUTPUT (all saved as PNG, 300 DPI, ready to paste into a report)
------
graphs/confusion_matrix.png
graphs/feature_importance.png
graphs/classification_metrics_by_class.png
graphs/roc_curves.png
graphs/precision_recall_curves.png
graphs/risk_class_distribution.png
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

MODEL_PATH = "model_xgboost.joblib"   # change to model_random_forest.joblib if preferred
DATA_PATH = "ml_training_dataset_v2.csv"
OUT_DIR = "graphs"
LABELS = ["Low", "Medium", "High", "Danger"]
COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#ef6c00", "Danger": "#c62828"}

os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "figure.dpi": 150})


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]
    label_encoder = saved.get("label_encoder")  # XGBoost models saved this; RF/logreg didn't
    print(f"Model expects {len(features)} features: {features}")

    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["event_time"], low_memory=False)

    for c in ["within_2km_narayani", "within_2km_rapti"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.upper().eq("TRUE").astype(int)

        n_river_nan = df["river_level_score"].isna().sum() if "river_level_score" in df.columns else 0
    if n_river_nan > 0:
        print(f"Filling {n_river_nan:,} NaN river_level_score values with 0.3 (the established "
              f"'unknown river condition' fallback -- see 04_train_models_v2.py) BEFORE dropna, "
              f"so these rows aren't silently discarded. Missing river data is disproportionately "
              f"concentrated on Rapti-side grids -- dropping instead of filling would systematically "
              f"under-represent Rapti, especially for the rare Danger class.")
        df["river_level_score"] = df["river_level_score"].fillna(0.3)

    df = df.dropna(subset=features + ["risk_class"])
    test = df[df["event_time"] >= "2026-01-01"]
    print(f"Test set: {len(test):,} rows")

    X_test = test[features]
    y_test = test["risk_class"]

    print("Running predictions ...")
    if label_encoder is not None:
        y_pred_enc = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_enc)
        y_proba = model.predict_proba(X_test)
        proba_classes = label_encoder.inverse_transform(np.arange(y_proba.shape[1]))
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        proba_classes = model.classes_

    # Reorder probability columns to match LABELS order for consistent plotting
    proba_df = pd.DataFrame(y_proba, columns=proba_classes)
    proba_df = proba_df[[c for c in LABELS if c in proba_df.columns]]

    # ------------------------------------------------------------------
    # 1. Confusion matrix
    # ------------------------------------------------------------------
    print("Generating confusion matrix ...")
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Flood Risk Classification")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Feature importance (tree-based models only)
    # ------------------------------------------------------------------
    if hasattr(model, "feature_importances_"):
        print("Generating feature importance chart ...")
        importances = pd.Series(model.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(importances.index, importances.values, color="#1565c0")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/feature_importance.png", dpi=300)
        plt.close(fig)
    else:
        print("Model has no feature_importances_ attribute (e.g. Logistic Regression) -- skipping.")

    # ------------------------------------------------------------------
    # 3. Per-class precision/recall/F1 bar chart
    # ------------------------------------------------------------------
    print("Generating per-class metrics chart ...")
    report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True)
    metrics_df = pd.DataFrame(report).T.loc[LABELS, ["precision", "recall", "f1-score"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(LABELS))
    width = 0.25
    ax.bar(x - width, metrics_df["precision"], width, label="Precision", color="#1565c0")
    ax.bar(x, metrics_df["recall"], width, label="Recall", color="#2e7d32")
    ax.bar(x + width, metrics_df["f1-score"], width, label="F1-score", color="#f9a825")
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall, F1-Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/classification_metrics_by_class.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. ROC curves (one-vs-rest, per class)
    # ------------------------------------------------------------------
    print("Generating ROC curves ...")
    y_test_bin = label_binarize(y_test, classes=LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, label in enumerate(LABELS):
        if label not in proba_df.columns:
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba_df[label])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})", color=COLORS[label], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/roc_curves.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Precision-Recall curves (more informative than ROC for imbalanced
    #    classes -- your Danger class is rare, PR curves show this better)
    # ------------------------------------------------------------------
    print("Generating precision-recall curves ...")
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, label in enumerate(LABELS):
        if label not in proba_df.columns:
            continue
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], proba_df[label])
        ap = average_precision_score(y_test_bin[:, i], proba_df[label])
        ax.plot(recall, precision, label=f"{label} (AP = {ap:.3f})", color=COLORS[label], linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/precision_recall_curves.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Risk class distribution -- actual vs predicted
    # ------------------------------------------------------------------
    print("Generating risk class distribution chart ...")
    actual_counts = y_test.value_counts().reindex(LABELS, fill_value=0)
    pred_counts = pd.Series(y_pred).value_counts().reindex(LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(LABELS))
    width = 0.35
    ax.bar(x - width/2, actual_counts.values, width, label="Actual", color="#455a64")
    ax.bar(x + width/2, pred_counts.values, width, label="Predicted", color="#1565c0")
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel("Count")
    ax.set_title("Risk Class Distribution — Actual vs Predicted (Test Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/risk_class_distribution.png", dpi=300)
    plt.close(fig)

    print(f"\nAll graphs saved to {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {OUT_DIR}/{f}")


if __name__ == "__main__":
    main()
