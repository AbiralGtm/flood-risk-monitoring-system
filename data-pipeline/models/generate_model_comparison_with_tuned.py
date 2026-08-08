"""
generate_model_comparison_with_tuned.py

PURPOSE
-------
Extends generate_model_comparison_graphs.py (LogReg vs RF vs XGBoost) to
add a 4th comparison point: XGBoost with the tuned Danger threshold
(model_xgboost_tuned.joblib), so you can see in one place whether
threshold tuning was actually worth it relative to all three original
models, not just relative to untuned XGBoost alone.

INPUT
------
model_logreg.joblib
model_random_forest.joblib
model_xgboost.joblib
model_xgboost_tuned.joblib     (from save_tuned_model.py)
ml_training_dataset_v2.csv

OUTPUT (comparison_with_tuned_graphs/)
------
accuracy_macroF1_comparison.png   -- 4 models
danger_class_comparison.png        -- 4 models
per_class_f1_comparison.png        -- 4 models
metrics_comparison_table.csv       -- full numeric table, all 4 models
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = "ml_training_dataset_v2.csv"
OUT_DIR = "comparison_with_tuned_graphs"
LABELS = ["Low", "Medium", "High", "Danger"]
MODEL_COLORS = {
    "Logistic Regression": "#78909c",
    "Random Forest": "#2e7d32",
    "XGBoost": "#1565c0",
    "XGBoost (Tuned)": "#c62828",
}

os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "figure.dpi": 150})


def predict_with_tuned_threshold(saved, X):
    """Same logic as save_tuned_model.py -- inlined here so this script
    doesn't depend on that file's location."""
    model = saved["model"]
    features = saved["features"]
    label_encoder = saved.get("label_encoder")
    threshold = saved["danger_threshold"]

    X = X[features]
    proba = model.predict_proba(X)
    if label_encoder is not None:
        proba_classes = label_encoder.inverse_transform(np.arange(proba.shape[1]))
    else:
        proba_classes = model.classes_
    proba_df = pd.DataFrame(proba, columns=proba_classes, index=X.index)

    other_classes_proba = proba_df.drop(columns=["Danger"])
    fallback_pred = other_classes_proba.idxmax(axis=1).values
    is_danger = proba_df["Danger"].values >= threshold
    return np.where(is_danger, "Danger", fallback_pred)


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["event_time"], low_memory=False)

    saved_xgb = joblib.load("model_xgboost.joblib")
    features = saved_xgb["features"]

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

    saved_lr = joblib.load("model_logreg.joblib")
    X_test_scaled = saved_lr["scaler"].transform(X_test) if saved_lr.get("scaler") is not None \
        else StandardScaler().fit_transform(X_test)

    results = {}

    # --- Logistic Regression ---
    print("Evaluating Logistic Regression ...")
    y_pred = saved_lr["model"].predict(X_test_scaled)
    results["Logistic Regression"] = y_pred

    # --- Random Forest ---
    print("Evaluating Random Forest ...")
    saved_rf = joblib.load("model_random_forest.joblib")
    y_pred = saved_rf["model"].predict(X_test)
    results["Random Forest"] = y_pred

    # --- XGBoost (untuned, argmax) ---
    print("Evaluating XGBoost (default argmax) ...")
    le = saved_xgb.get("label_encoder")
    if le is not None:
        y_pred = le.inverse_transform(saved_xgb["model"].predict(X_test))
    else:
        y_pred = saved_xgb["model"].predict(X_test)
    results["XGBoost"] = y_pred

    # --- XGBoost (tuned threshold) ---
    print("Evaluating XGBoost (tuned threshold) ...")
    saved_tuned = joblib.load("model_xgboost_tuned.joblib")
    y_pred = predict_with_tuned_threshold(saved_tuned, X_test)
    results["XGBoost (Tuned)"] = y_pred
    print(f"  (tuned threshold = {saved_tuned['danger_threshold']:.4f}, "
          f"method = {saved_tuned['threshold_method']})")

    names = list(results.keys())
    reports = {n: classification_report(y_test, results[n], labels=LABELS, output_dict=True) for n in names}
    accs = {n: accuracy_score(y_test, results[n]) for n in names}
    f1s = {n: f1_score(y_test, results[n], average="macro") for n in names}

    # ------------------------------------------------------------------
    # 1. Overall accuracy + macro F1, 4 models
    # ------------------------------------------------------------------
    print("Generating overall comparison ...")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, [accs[n] for n in names], width, label="Accuracy", color="#1565c0")
    ax.bar(x + width/2, [f1s[n] for n in names], width, label="Macro F1", color="#f9a825")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall Model Comparison (including tuned threshold)")
    for i, n in enumerate(names):
        ax.text(i - width/2, accs[n] + 0.02, f"{accs[n]:.3f}", ha="center", fontsize=9)
        ax.text(i + width/2, f1s[n] + 0.02, f"{f1s[n]:.3f}", ha="center", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/accuracy_macroF1_comparison.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Danger-class comparison, 4 models
    # ------------------------------------------------------------------
    print("Generating Danger-class comparison ...")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3)
    width = 0.2
    metric_names = ["precision", "recall", "f1-score"]
    for i, n in enumerate(names):
        vals = [reports[n]["Danger"][m] for m in metric_names]
        ax.bar(x + (i - 1.5) * width, vals, width, label=n, color=MODEL_COLORS[n])
    ax.set_xticks(x); ax.set_xticklabels(["Precision", "Recall", "F1-score"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Danger-Class Performance (including tuned threshold)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/danger_class_comparison.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Per-class F1, 4 models
    # ------------------------------------------------------------------
    print("Generating per-class F1 comparison ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(LABELS))
    width = 0.2
    for i, n in enumerate(names):
        vals = [reports[n][label]["f1-score"] for label in LABELS]
        ax.bar(x + (i - 1.5) * width, vals, width, label=n, color=MODEL_COLORS[n])
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-Class F1-Score (including tuned threshold)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/per_class_f1_comparison.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Full numeric table
    # ------------------------------------------------------------------
    rows = []
    for n in names:
        row = {"model": n, "accuracy": accs[n], "macro_f1": f1s[n]}
        for label in LABELS:
            row[f"{label}_precision"] = reports[n][label]["precision"]
            row[f"{label}_recall"] = reports[n][label]["recall"]
            row[f"{label}_f1"] = reports[n][label]["f1-score"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(f"{OUT_DIR}/metrics_comparison_table.csv", index=False)

    print(f"\nAll outputs saved to {OUT_DIR}/")
    print("\n--- Summary (accuracy, macro F1, Danger precision/recall) ---")
    summary = table[["model", "accuracy", "macro_f1", "Danger_precision", "Danger_recall"]]
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()