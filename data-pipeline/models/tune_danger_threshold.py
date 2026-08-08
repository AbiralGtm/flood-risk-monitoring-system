"""
tune_danger_threshold.py

PURPOSE
-------
Same core idea as before: your model's default argmax decision rule
gives Danger precision/recall of ~0.50/~0.999. This sweeps possible
Danger probability thresholds and finds better operating points, then
gives a FULL before/after comparison across ALL FOUR risk classes (not
just Danger), so you can see the complete picture of what tuning the
threshold changes -- including any side effects on Low/Medium/High.

CHANGES FROM THE PREVIOUS VERSION
-----------------------------------
1. Full multi-class before/after comparison: overall accuracy/macro F1,
   plus a per-class precision/recall/F1 grouped bar chart (default vs
   tuned), not just the Danger row.
2. All outputs go into their own dedicated subfolder
   (threshold_comparison_graphs/) so nothing here overwrites the output
   of generate_model_graphs.py (graphs/) or
   generate_model_comparison_graphs.py (comparison_graphs/).
3. Font set to Times New Roman at 12pt globally. Times New Roman is
   the standard metric-compatible substitute for Times New Roman on
   Linux (same character widths/spacing, near-identical appearance) --
   actual Times New Roman is a proprietary Microsoft font not present
   on this system. If you open these PNGs on a machine that DOES have
   Times New Roman installed, they'll still look the way you expect
   since these are pre-rendered images, not editable text.

OUTPUT (in threshold_comparison_graphs/)
------------------------------------------
threshold_tuning_curve.png              -- precision/recall/F1 vs threshold
confusion_matrix_before_after.png       -- side-by-side, default vs tuned
overall_metrics_before_after.png        -- accuracy + macro F1, default vs tuned
per_class_metrics_before_after.png      -- precision/recall/F1, ALL 4 classes,
                                            default vs tuned, grouped bars
metrics_comparison_table.csv            -- full numeric table, both operating
                                            points, all classes -- for pasting
                                            directly into your report
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix, classification_report,
    accuracy_score, f1_score
)

MODEL_PATH = "model_xgboost.joblib"
DATA_PATH = "ml_training_dataset_v2.csv"
OUT_DIR = "threshold_comparison_graphs"
LABELS = ["Low", "Medium", "High", "Danger"]
MIN_RECALL_FLOOR = 0.95

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Font: Times New Roman (Times New Roman metric-compatible substitute),
# 12pt, applied globally so every plot in this script matches.
# ----------------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 150


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]
    label_encoder = saved.get("label_encoder")

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
    X_test = test[features]
    y_test = test["risk_class"]
    print(f"Test set: {len(test):,} rows")

    if label_encoder is not None:
        y_pred_default = label_encoder.inverse_transform(model.predict(X_test))
        proba = model.predict_proba(X_test)
        proba_classes = label_encoder.inverse_transform(np.arange(proba.shape[1]))
    else:
        y_pred_default = model.predict(X_test)
        proba = model.predict_proba(X_test)
        proba_classes = model.classes_

    proba_df = pd.DataFrame(proba, columns=proba_classes)
    danger_proba = proba_df["Danger"].values
    y_test_danger = (y_test == "Danger").astype(int).values

    # ------------------------------------------------------------------
    # Sweep thresholds
    # ------------------------------------------------------------------
    print("Sweeping thresholds ...")
    precisions, recalls, thresholds = precision_recall_curve(y_test_danger, danger_proba)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)

    best_f1_idx = np.argmax(f1s)
    best_f1_threshold = thresholds[best_f1_idx]

    valid_recall_mask = recalls[:-1] >= MIN_RECALL_FLOOR
    if valid_recall_mask.any():
        valid_precisions = np.where(valid_recall_mask, precisions[:-1], -1)
        high_recall_idx = np.argmax(valid_precisions)
        high_recall_threshold = thresholds[high_recall_idx]
    else:
        high_recall_idx = None
        high_recall_threshold = None
        print(f"WARNING: no threshold achieves recall >= {MIN_RECALL_FLOOR}")

    chosen_threshold = high_recall_threshold if high_recall_threshold is not None else best_f1_threshold
    chosen_label = f"Tuned (recall floor >= {MIN_RECALL_FLOOR})" if high_recall_threshold is not None else "Tuned (best F1)"

    print(f"\nChosen tuned threshold: {chosen_threshold:.3f}")

    # ------------------------------------------------------------------
    # Build tuned predictions
    # ------------------------------------------------------------------
    other_classes_proba = proba_df.drop(columns=["Danger"])
    fallback_pred = other_classes_proba.idxmax(axis=1).values
    is_danger_tuned = danger_proba >= chosen_threshold
    y_pred_tuned = np.where(is_danger_tuned, "Danger", fallback_pred)

    # ------------------------------------------------------------------
    # Plot 1: threshold sweep curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, precisions[:-1], label="Precision", color="#1565c0", linewidth=2)
    ax.plot(thresholds, recalls[:-1], label="Recall", color="#2e7d32", linewidth=2)
    ax.plot(thresholds, f1s, label="F1", color="#f9a825", linewidth=2, linestyle="--")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.6, label="Rough default (~0.5)")
    ax.axvline(best_f1_threshold, color="#c62828", linestyle="-", alpha=0.8,
               label=f"Best F1 @ {best_f1_threshold:.2f}")
    if high_recall_threshold is not None:
        ax.axvline(high_recall_threshold, color="#6a1b9a", linestyle="-", alpha=0.8,
                   label=f"High-recall-floor @ {high_recall_threshold:.2f}")
    ax.set_xlabel("Danger probability threshold")
    ax.set_ylabel("Score")
    ax.set_title("Danger-Class Threshold Tuning")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/threshold_tuning_curve.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: confusion matrices, default vs tuned
    # ------------------------------------------------------------------
    cm_before = confusion_matrix(y_test, y_pred_default, labels=LABELS)
    cm_after = confusion_matrix(y_test, y_pred_tuned, labels=LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, cm, title in zip(axes, [cm_before, cm_after],
                               ["Default (argmax)", f"{chosen_label}, thresh={chosen_threshold:.2f}"]):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
        ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", color=color, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/confusion_matrix_before_after.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 3: overall accuracy + macro F1, before vs after
    # ------------------------------------------------------------------
    acc_before, acc_after = accuracy_score(y_test, y_pred_default), accuracy_score(y_test, y_pred_tuned)
    f1_before, f1_after = f1_score(y_test, y_pred_default, average="macro"), f1_score(y_test, y_pred_tuned, average="macro")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [acc_before, f1_before], width, label="Default (argmax)", color="#78909c")
    ax.bar(x + width/2, [acc_after, f1_after], width, label=chosen_label, color="#1565c0")
    ax.set_xticks(x); ax.set_xticklabels(["Accuracy", "Macro F1"])
    ax.set_ylim(0, 1.05)
    for i, (b, a) in enumerate(zip([acc_before, f1_before], [acc_after, f1_after])):
        ax.text(i - width/2, b + 0.02, f"{b:.3f}", ha="center", fontsize=10)
        ax.text(i + width/2, a + 0.02, f"{a:.3f}", ha="center", fontsize=10)
    ax.set_title("Overall Metrics — Default vs Tuned Threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/overall_metrics_before_after.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 4: per-class precision/recall/F1, before vs after, ALL classes
    # ------------------------------------------------------------------
    report_before = classification_report(y_test, y_pred_default, labels=LABELS, output_dict=True)
    report_after = classification_report(y_test, y_pred_tuned, labels=LABELS, output_dict=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    metric_names = ["precision", "recall", "f1-score"]
    metric_titles = ["Precision", "Recall", "F1-score"]
    for ax, metric, title in zip(axes, metric_names, metric_titles):
        before_vals = [report_before[l][metric] for l in LABELS]
        after_vals = [report_after[l][metric] for l in LABELS]
        x = np.arange(len(LABELS))
        width = 0.35
        ax.bar(x - width/2, before_vals, width, label="Default", color="#78909c")
        ax.bar(x + width/2, after_vals, width, label="Tuned", color="#1565c0")
        ax.set_xticks(x); ax.set_xticklabels(LABELS)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[0].legend()
    fig.suptitle("Per-Class Metrics — Default vs Tuned Threshold")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/per_class_metrics_before_after.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Full numeric comparison table -> CSV, for pasting into the report
    # ------------------------------------------------------------------
    rows = []
    for label in LABELS:
        rows.append({
            "class": label,
            "precision_default": report_before[label]["precision"],
            "recall_default": report_before[label]["recall"],
            "f1_default": report_before[label]["f1-score"],
            "precision_tuned": report_after[label]["precision"],
            "recall_tuned": report_after[label]["recall"],
            "f1_tuned": report_after[label]["f1-score"],
        })
    rows.append({
        "class": "OVERALL (accuracy/macro F1)",
        "precision_default": acc_before, "recall_default": np.nan, "f1_default": f1_before,
        "precision_tuned": acc_after, "recall_tuned": np.nan, "f1_tuned": f1_after,
    })
    comparison_table = pd.DataFrame(rows)
    comparison_table.to_csv(f"{OUT_DIR}/metrics_comparison_table.csv", index=False)

    print(f"\nAll outputs saved to {OUT_DIR}/ (does not overwrite graphs/ or comparison_graphs/)")
    print("\n--- Full comparison table ---")
    print(comparison_table.round(4).to_string(index=False))
    print(f"\nChosen tuned threshold: {chosen_threshold:.3f} ({chosen_label})")
    print("If you adopt this threshold, apply it consistently in your backend's")
    print("/predict endpoint too, not just in this evaluation script.")


if __name__ == "__main__":
    main()
