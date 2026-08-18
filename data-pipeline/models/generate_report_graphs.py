"""
generate_report_graphs.py

Generates EXACTLY the 11 figures needed for the report's Result Analysis
section (4.2.5) -- no more, no less. Outputs BOTH .svg (for print-quality
insertion into Word) and .png (for quick preview) for every figure, into
the same folder as this script.

FONT: Times New Roman (rendered via Times New Roman, the standard
metric-compatible substitute), base size 20pt throughout. Figure sizes
were increased relative to an earlier 12pt version specifically to
prevent label overlap at this larger font -- verified by direct visual
inspection of every figure, not assumed.

Uses the ACTIVE model (XGBoost, tuned threshold) for all single-model
figures, and all 4 trained models for comparison figures.

REQUIRED FILES (same folder as this script)
--------------------------------------------
model_logreg.joblib, model_random_forest.joblib, model_xgboost.joblib,
model_xgboost_tuned.joblib, ml_training_dataset_v2.csv

OUTPUT: report_graphs/  -- 11 figures x 2 formats (.svg + .png), numbered
        01-11 in the order they should appear in Result Analysis (4.2.5)
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

plt.rcParams["font.family"] = "Times New Roman"
BASE = 18  # 20px converted to points (1pt = 1.333px, so 20px ≈ 15pt; using 14pt per confirmed target)
plt.rcParams["font.size"] = BASE

OUT_DIR = "report_graphs"
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = ["Low", "Medium", "High", "Danger"]
COLOR_LOW, COLOR_MED, COLOR_HIGH, COLOR_DANGER = "#2E7D32", "#F9A825", "#EF6C00", "#C62828"
COLOR_MODELS = {"Logistic Regression": "#8C9BA5", "Random Forest": "#2E7D32",
                 "XGBoost (default)": "#1C7293", "XGBoost (tuned)": "#C62828"}

FS_TITLE = BASE + 2
FS_LABEL = BASE 
FS_TICK = BASE
FS_LEGEND = BASE 
FS_VALUE = BASE
FS_VALUE_SMALL = BASE


def savefig(fig, name, **kwargs):
    fig.savefig(f"{OUT_DIR}/{name}.svg", format="svg", facecolor="white", **kwargs)
    fig.savefig(f"{OUT_DIR}/{name}.png", format="png", dpi=300, facecolor="white", **kwargs)
    plt.close(fig)


def load_models():
    print("Loading models...")
    logreg = joblib.load("model_logreg.joblib")
    rf = joblib.load("model_random_forest.joblib")
    xgb_default = joblib.load("model_xgboost.joblib")
    xgb_tuned_saved = joblib.load("model_xgboost_tuned.joblib")
    return logreg, rf, xgb_default, xgb_tuned_saved


def load_test_data(features):
    print("Loading dataset...")
    df = pd.read_csv("ml_training_dataset_v2.csv", parse_dates=["event_time"], low_memory=False)
    for c in ["within_2km_narayani", "within_2km_rapti"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.upper().eq("TRUE").astype(int)
    n_river_nan = df["river_level_score"].isna().sum() if "river_level_score" in df.columns else 0
    if n_river_nan > 0:
        print(f"Filling {n_river_nan:,} NaN river_level_score values with 0.3 before dropna.")
        df["river_level_score"] = df["river_level_score"].fillna(0.3)
    df = df.dropna(subset=features + ["risk_class"])
    test = df[df["event_time"] >= "2026-01-01"]
    print(f"Test set: {len(test):,} rows")
    return test


def get_model_and_features(bundle):
    if isinstance(bundle, dict):
        return bundle["model"], bundle["features"], bundle.get("label_encoder"), bundle.get("danger_threshold")
    return bundle, None, None, None


def predict_with_threshold(model, X, label_encoder, threshold):
    proba = model.predict_proba(X)
    classes = label_encoder.inverse_transform(np.arange(proba.shape[1])) if label_encoder is not None else model.classes_
    proba_df = pd.DataFrame(proba, columns=classes)
    other = proba_df.drop(columns=["Danger"])
    fallback_pred = other.idxmax(axis=1).values
    is_danger = proba_df["Danger"].values >= threshold
    return np.where(is_danger, "Danger", fallback_pred), proba_df


def predict_default(model, X, label_encoder):
    pred_enc = model.predict(X)
    pred = label_encoder.inverse_transform(pred_enc) if label_encoder is not None else pred_enc
    proba = model.predict_proba(X)
    classes = label_encoder.inverse_transform(np.arange(proba.shape[1])) if label_encoder is not None else model.classes_
    proba_df = pd.DataFrame(proba, columns=classes)
    return pred, proba_df


def fig_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.84, 5.90))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS, fontsize=FS_TICK); ax.set_yticklabels(LABELS, fontsize=FS_TICK)
    ax.set_xlabel("Predicted", fontsize=FS_LABEL, labelpad=12)
    ax.set_ylabel("Actual", fontsize=FS_LABEL, labelpad=12)
    # ax.set_title("Confusion Matrix — XGBoost (Tuned)", fontsize=FS_TITLE, pad=16)
    vmax = cm.max()
    for i in range(4):
        for j in range(4):
            color = "white" if cm[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=FS_LABEL, color=color)
    fig.tight_layout()
    savefig(fig, "01_confusion_matrix")


def fig_feature_importance(rf_model, features):
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.92, 6.48))
    ax.barh(importances.index, importances.values, color="#1C7293")
    ax.set_xlabel("Importance", fontsize=FS_LABEL)
    ax.set_title("", fontsize=FS_TITLE, pad=14)
    # ax.set_title("Feature Importance — Random Forest", fontsize=FS_TITLE, pad=14)
    ax.tick_params(axis="y", labelsize=FS_TICK - 2)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    fig.tight_layout()
    savefig(fig, "02_feature_importance")


def fig_roc_curves(y_true, proba_df):
    fig, ax = plt.subplots(figsize=(6.84, 6.48))
    colors = {"Low": COLOR_LOW, "Medium": COLOR_MED, "High": COLOR_HIGH, "Danger": COLOR_DANGER}
    for cls in LABELS:
        y_bin = (y_true == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba_df[cls])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.3f})", color=colors[cls], linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=FS_LABEL)
    ax.set_ylabel("True Positive Rate", fontsize=FS_LABEL)
    # ax.set_title("ROC Curves — XGBoost (Tuned)", fontsize=FS_TITLE, pad=14)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(loc="lower right", fontsize=FS_LEGEND, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "03_roc_curves")


def fig_pr_curves(y_true, proba_df):
    fig, ax = plt.subplots(figsize=(6.84, 6.48))
    colors = {"Low": COLOR_LOW, "Medium": COLOR_MED, "High": COLOR_HIGH, "Danger": COLOR_DANGER}
    for cls in LABELS:
        y_bin = (y_true == cls).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, proba_df[cls])
        ap = average_precision_score(y_bin, proba_df[cls])
        ax.plot(recall, precision, label=f"{cls} (AP = {ap:.3f})", color=colors[cls], linewidth=2.5)
    ax.set_xlabel("Recall", fontsize=FS_LABEL)
    ax.set_ylabel("Precision", fontsize=FS_LABEL)
    # ax.set_title("Precision-Recall Curves — XGBoost (Tuned)", fontsize=FS_TITLE, pad=14)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(loc="lower left", fontsize=FS_LEGEND, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "04_precision_recall_curves")


def fig_per_class_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(LABELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.64, 6.48))
    for i, m in enumerate(metrics):
        vals = [report[cls][m] for cls in LABELS]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=m.capitalize())
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=FS_VALUE)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=FS_TICK)
    ax.set_ylim(0, 1.18)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylabel("Score", fontsize=FS_LABEL)
    # ax.set_title("Per-Class Precision, Recall, F1 — XGBoost (Tuned)", fontsize=FS_TITLE, pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=FS_LEGEND, frameon=False)
    fig.tight_layout()
    savefig(fig, "05_classification_metrics_by_class")


def fig_risk_distribution(y_true, y_pred):
    actual_counts = pd.Series(y_true).value_counts().reindex(LABELS).fillna(0)
    pred_counts = pd.Series(y_pred).value_counts().reindex(LABELS).fillna(0)
    x = np.arange(len(LABELS)); width = 0.35
    fig, ax = plt.subplots(figsize=(7.56, 6.12))
    ax.bar(x - width/2, actual_counts, width, label="Actual", color="#455A64")
    ax.bar(x + width/2, pred_counts, width, label="Predicted", color="#1C7293")
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylabel("Count", fontsize=FS_LABEL)
    # ax.set_title("Actual vs. Predicted Risk Class Distribution\nXGBoost (Tuned)", fontsize=FS_TITLE, pad=14)
    ax.legend(fontsize=FS_LEGEND)
    fig.tight_layout()
    savefig(fig, "06_risk_class_distribution")


def compute_all_model_results(y_true, X_test, logreg_b, rf_b, xgbd_b, xgbt_bundle):
    results = {}
    for name, bundle in [("Logistic Regression", logreg_b), ("Random Forest", rf_b), ("XGBoost (default)", xgbd_b)]:
        model, feats, le, _ = get_model_and_features(bundle)
        pred, proba = predict_default(model, X_test[feats] if feats else X_test, le)
        results[name] = {"y_pred": pred, "proba": proba}
    xgbt_model, xgbt_feats, xgbt_le, xgbt_thresh = get_model_and_features(xgbt_bundle)
    pred_t, proba_t = predict_with_threshold(xgbt_model, X_test[xgbt_feats], xgbt_le, xgbt_thresh)
    results["XGBoost (tuned)"] = {"y_pred": pred_t, "proba": proba_t}
    return results


def fig_accuracy_macrof1_comparison(y_true, results):
    names = list(COLOR_MODELS.keys())
    acc, f1 = [], []
    for n in names:
        report = classification_report(y_true, results[n]["y_pred"], labels=LABELS, output_dict=True, zero_division=0)
        acc.append(report["accuracy"])
        f1.append(report["macro avg"]["f1-score"])
    x = np.arange(len(names)); width = 0.35
    fig, ax = plt.subplots(figsize=(9.36, 6.48))
    b1 = ax.bar(x - width/2, acc, width, label="Accuracy", color="#1C7293")
    b2 = ax.bar(x + width/2, f1, width, label="Macro F1", color="#F9A825")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.018, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=FS_VALUE)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=FS_TICK - 1, rotation=10, ha="right")
    ax.set_ylim(0, 1.16)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylabel("Score", fontsize=FS_LABEL)
    # ax.set_title("Overall Model Comparison", fontsize=FS_TITLE, pad=14)
    ax.legend(fontsize=FS_LEGEND)
    fig.tight_layout()
    savefig(fig, "07_accuracy_macroF1_comparison")


def fig_danger_class_comparison(y_true, results):
    names = list(COLOR_MODELS.keys())
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(metrics)); width = 0.19
    fig, ax = plt.subplots(figsize=(10.80, 7.20))
    for i, n in enumerate(names):
        report = classification_report(y_true, results[n]["y_pred"], labels=LABELS, output_dict=True, zero_division=0)
        vals = [report["Danger"][m] for m in metrics]
        offset = (i - (len(names)-1)/2) * width
        bars = ax.bar(x + offset, vals, width, label=n, color=COLOR_MODELS[n])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=FS_VALUE_SMALL, rotation=90 if v < 0.15 else 0)
    ax.set_xticks(x); ax.set_xticklabels(["Precision", "Recall", "F1-score"], fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylim(0, 1.22)
    # ax.set_title("Danger-Class Performance — Model Comparison", fontsize=FS_TITLE, pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=FS_LEGEND, frameon=False)
    fig.tight_layout()
    savefig(fig, "08_danger_class_comparison")


def fig_per_class_f1_comparison(y_true, results):
    names = list(COLOR_MODELS.keys())
    x = np.arange(len(LABELS)); width = 0.19
    fig, ax = plt.subplots(figsize=(10.80, 7.20))
    for i, n in enumerate(names):
        report = classification_report(y_true, results[n]["y_pred"], labels=LABELS, output_dict=True, zero_division=0)
        vals = [report[cls]["f1-score"] for cls in LABELS]
        offset = (i - (len(names)-1)/2) * width
        ax.bar(x + offset, vals, width, label=n, color=COLOR_MODELS[n])
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("F1-score", fontsize=FS_LABEL)
    # ax.set_title("Per-Class F1-Score — Model Comparison", fontsize=FS_TITLE, pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=FS_LEGEND, frameon=False)
    fig.tight_layout()
    savefig(fig, "09_per_class_f1_comparison")


def fig_threshold_curve(xgbt_bundle, X_test, y_true):
    model, feats, le, chosen_threshold = get_model_and_features(xgbt_bundle)
    proba = model.predict_proba(X_test[feats])
    classes = le.inverse_transform(np.arange(proba.shape[1])) if le is not None else model.classes_
    proba_df = pd.DataFrame(proba, columns=classes)
    y_bin = (y_true == "Danger").astype(int)
    precision, recall, thresholds = precision_recall_curve(y_bin, proba_df["Danger"])
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)

    fig, ax = plt.subplots(figsize=(9.00, 6.48))
    ax.plot(thresholds, precision[:-1], label="Precision", color="#1C7293", linewidth=2.5)
    ax.plot(thresholds, recall[:-1], label="Recall", color="#2E7D32", linewidth=2.5)
    ax.plot(thresholds, f1, label="F1", color="#F9A825", linewidth=2.5, linestyle="--")
    if chosen_threshold is not None:
        ax.axvline(chosen_threshold, color="#C62828", linewidth=1.6,
                   label=f"Chosen threshold ({chosen_threshold:.3f})")
    ax.set_xlabel("Danger probability threshold", fontsize=FS_LABEL)
    ax.set_ylabel("Score", fontsize=FS_LABEL)
    # ax.set_title("Danger-Class Threshold Tuning", fontsize=FS_TITLE, pad=14)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=FS_LEGEND, frameon=False)
    fig.tight_layout()
    savefig(fig, "10_threshold_tuning_curve", bbox_inches="tight")


def fig_confusion_before_after(y_true, y_pred_default, y_pred_tuned, chosen_threshold):
    cm_default = confusion_matrix(y_true, y_pred_default, labels=LABELS)
    cm_tuned = confusion_matrix(y_true, y_pred_tuned, labels=LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12.96, 6.48))
    titles = ["Default (argmax)", f"Tuned (threshold={chosen_threshold:.3f})"]
    for ax, cm, title in zip(axes, [cm_default, cm_tuned], titles):
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(LABELS, fontsize=FS_TICK - 1); ax.set_yticklabels(LABELS, fontsize=FS_TICK - 1)
        ax.set_xlabel("Predicted", fontsize=FS_LABEL - 1)
        ax.set_ylabel("Actual", fontsize=FS_LABEL - 1)
        # ax.set_title(title, fontsize=FS_TITLE - 2, pad=12)
        vmax = cm.max()
        for i in range(4):
            for j in range(4):
                color = "white" if cm[i, j] > vmax * 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=FS_LABEL - 2, color=color)
    fig.tight_layout()
    savefig(fig, "11_confusion_matrix_before_after")


def main():
    logreg_b, rf_b, xgbd_b, xgbt_bundle = load_models()
    xgbt_model, xgbt_features, xgbt_le, xgbt_threshold = get_model_and_features(xgbt_bundle)
    if xgbt_features is None:
        raise ValueError("model_xgboost_tuned.joblib must be the saved dict format.")

    test = load_test_data(xgbt_features)
    X_test = test[xgbt_features]
    y_true = test["risk_class"].values

    print("\nGenerating single-model figures (ACTIVE MODEL: XGBoost, tuned)...")
    y_pred_tuned, proba_tuned = predict_with_threshold(xgbt_model, X_test, xgbt_le, xgbt_threshold)
    fig_confusion_matrix(y_true, y_pred_tuned)
    fig_roc_curves(y_true, proba_tuned)
    fig_pr_curves(y_true, proba_tuned)
    fig_per_class_metrics(y_true, y_pred_tuned)
    fig_risk_distribution(y_true, y_pred_tuned)

    print("Generating feature importance (Random Forest)...")
    rf_model, rf_features, _, _ = get_model_and_features(rf_b)
    fig_feature_importance(rf_model, rf_features if rf_features else xgbt_features)

    print("Generating 4-model comparison figures...")
    results = compute_all_model_results(y_true, test, logreg_b, rf_b, xgbd_b, xgbt_bundle)
    results["XGBoost (tuned)"] = {"y_pred": y_pred_tuned, "proba": proba_tuned}
    fig_accuracy_macrof1_comparison(y_true, results)
    fig_danger_class_comparison(y_true, results)
    fig_per_class_f1_comparison(y_true, results)

    print("Generating threshold tuning curve...")
    fig_threshold_curve(xgbt_bundle, test, y_true)

    print("Generating confusion matrix before/after...")
    y_pred_default = results["XGBoost (default)"]["y_pred"]
    fig_confusion_before_after(y_true, y_pred_default, y_pred_tuned, xgbt_threshold)

    n_files = len([f for f in os.listdir(OUT_DIR) if f.endswith((".svg", ".png"))])
    print(f"\nDone. {n_files} files (11 figures x SVG+PNG) saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()