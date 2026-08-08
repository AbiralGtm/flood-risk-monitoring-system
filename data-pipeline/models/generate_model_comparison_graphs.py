"""
generate_model_comparison_graphs.py

PURPOSE
-------
Compares all three models from your proposal's Table 3.3 (Logistic
Regression baseline, Random Forest, XGBoost) side by side -- the single-
model script (generate_model_graphs.py) only covers one model at a time
and doesn't show whether XGBoost actually earned its place as your final
choice over the other two.

INPUT
------
model_logreg.joblib
model_random_forest.joblib
model_xgboost.joblib
ml_training_dataset_v2.csv

OUTPUT (all PNG, 300 DPI)
------
comparison_graphs/accuracy_macroF1_comparison.png   -- overall metrics, 3 models
comparison_graphs/danger_class_comparison.png        -- Danger-class precision/recall/F1, 3 models
comparison_graphs/per_class_f1_comparison.png        -- F1 per risk class, 3 models grouped
comparison_graphs/roc_danger_comparison.png          -- Danger-class ROC, 3 models overlaid
comparison_graphs/pr_danger_comparison.png           -- Danger-class PR curve, 3 models overlaid
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_curve, auc, precision_recall_curve,
    average_precision_score, accuracy_score, f1_score
)
from sklearn.preprocessing import StandardScaler, label_binarize

DATA_PATH = "ml_training_dataset_v2.csv"
OUT_DIR = "comparison_graphs"
LABELS = ["Low", "Medium", "High", "Danger"]
MODEL_COLORS = {"Logistic Regression": "#78909c", "Random Forest": "#2e7d32", "XGBoost": "#1565c0"}

os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "figure.dpi": 150})


def load_model_predictions(path, X_test, X_test_scaled=None):
    """Returns (y_pred, proba_df) for one saved model, handling the
    label-encoder difference between XGBoost and the other two."""
    saved = joblib.load(path)
    model = saved["model"]
    label_encoder = saved.get("label_encoder")
    scaler = saved.get("scaler")

    X_input = X_test_scaled if scaler is not None else X_test

    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(model.predict(X_input))
        proba = model.predict_proba(X_input)
        proba_classes = label_encoder.inverse_transform(np.arange(proba.shape[1]))
    else:
        y_pred = model.predict(X_input)
        proba = model.predict_proba(X_input)
        proba_classes = model.classes_

    proba_df = pd.DataFrame(proba, columns=proba_classes)
    proba_df = proba_df[[c for c in LABELS if c in proba_df.columns]]
    return y_pred, proba_df


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["event_time"], low_memory=False)

    saved_xgb = joblib.load("model_xgboost.joblib")
    features = saved_xgb["features"]  # use the model's own saved feature list

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
    y_test_bin = label_binarize(y_test, classes=LABELS)

    # Logistic Regression needs scaled input -- fit scaler on this same
    # test set's features as a stand-in (ideally you'd reuse the exact
    # training-fitted scaler; if model_logreg.joblib includes one, use it)
    saved_lr = joblib.load("model_logreg.joblib")
    if saved_lr.get("scaler") is not None:
        X_test_scaled = saved_lr["scaler"].transform(X_test)
    else:
        X_test_scaled = StandardScaler().fit_transform(X_test)

    results = {}
    model_files = {
        "Logistic Regression": "model_logreg.joblib",
        "Random Forest": "model_random_forest.joblib",
        "XGBoost": "model_xgboost.joblib",
    }
    for name, path in model_files.items():
        print(f"Evaluating {name} ...")
        y_pred, proba_df = load_model_predictions(path, X_test, X_test_scaled)
        report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True)
        results[name] = {
            "y_pred": y_pred,
            "proba_df": proba_df,
            "report": report,
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
        }

    # ------------------------------------------------------------------
    # 1. Overall accuracy + macro F1, 3 models
    # ------------------------------------------------------------------
    print("Generating overall accuracy/macro-F1 comparison ...")
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(model_files.keys())
    x = np.arange(len(names))
    width = 0.35
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["macro_f1"] for n in names]
    ax.bar(x - width/2, accs, width, label="Accuracy", color="#1565c0")
    ax.bar(x + width/2, f1s, width, label="Macro F1", color="#f9a825")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall Model Comparison")
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - width/2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + width/2, f + 0.02, f"{f:.3f}", ha="center", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/accuracy_macroF1_comparison.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Danger-class precision/recall/F1, 3 models
    # ------------------------------------------------------------------
    print("Generating Danger-class comparison ...")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)  # precision, recall, f1
    width = 0.25
    metric_names = ["precision", "recall", "f1-score"]
    for i, name in enumerate(names):
        vals = [results[name]["report"]["Danger"][m] for m in metric_names]
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=MODEL_COLORS[name])
    ax.set_xticks(x); ax.set_xticklabels(["Precision", "Recall", "F1-score"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Danger-Class Performance — Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/danger_class_comparison.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. F1 per class, 3 models grouped
    # ------------------------------------------------------------------
    print("Generating per-class F1 comparison ...")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(LABELS))
    width = 0.25
    for i, name in enumerate(names):
        vals = [results[name]["report"][label]["f1-score"] for label in LABELS]
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=MODEL_COLORS[name])
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-Class F1-Score — Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/per_class_f1_comparison.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Danger-class ROC, 3 models overlaid (the class that matters most)
    # ------------------------------------------------------------------
    print("Generating Danger-class ROC comparison ...")
    danger_idx = LABELS.index("Danger")
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in names:
        proba_df = results[name]["proba_df"]
        if "Danger" not in proba_df.columns:
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, danger_idx], proba_df["Danger"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=MODEL_COLORS[name], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Danger-Class ROC — Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/roc_danger_comparison.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Danger-class PR curve, 3 models overlaid
    # ------------------------------------------------------------------
    print("Generating Danger-class PR comparison ...")
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in names:
        proba_df = results[name]["proba_df"]
        if "Danger" not in proba_df.columns:
            continue
        precision, recall, _ = precision_recall_curve(y_test_bin[:, danger_idx], proba_df["Danger"])
        ap = average_precision_score(y_test_bin[:, danger_idx], proba_df["Danger"])
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})", color=MODEL_COLORS[name], linewidth=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Danger-Class Precision-Recall — Model Comparison")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/pr_danger_comparison.png", dpi=300)
    plt.close(fig)

    print(f"\nAll comparison graphs saved to {OUT_DIR}/")
    print("\nSummary table:")
    summary = pd.DataFrame({
        name: {
            "Accuracy": results[name]["accuracy"],
            "Macro F1": results[name]["macro_f1"],
            "Danger Precision": results[name]["report"]["Danger"]["precision"],
            "Danger Recall": results[name]["report"]["Danger"]["recall"],
            "Danger F1": results[name]["report"]["Danger"]["f1-score"],
        }
        for name in names
    }).T
    print(summary.round(4))


if __name__ == "__main__":
    main()
