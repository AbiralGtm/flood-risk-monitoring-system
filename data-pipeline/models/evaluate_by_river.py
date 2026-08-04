"""
evaluate_by_river.py

PURPOSE
-------
Tests whether training one combined model on both Narayani and Rapti
grids produces river-fair results, or whether it's quietly worse for
Rapti specifically -- given the known data asymmetry (Rapti has more
missing river-level readings even in real observed data, and NO
calibrated historical stage curve at all, unlike Narayani/Devghat).

This is the same principle as the earlier data_source (observed vs.
reconstructed) stratified check -- split the SAME test set by river,
and compute the SAME metrics separately for each, rather than trusting
one blended number to represent both rivers equally.

INPUT
------
model_xgboost.joblib
ml_training_dataset_v2.csv

OUTPUT
------
Prints per-river classification reports + a comparison table.
river_comparison_metrics.csv -- for pasting into your report.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, f1_score

MODEL_PATH = "model_xgboost.joblib"
DATA_PATH = "ml_training_dataset_v2.csv"
LABELS = ["Low", "Medium", "High", "Danger"]


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

    df = df.dropna(subset=features + ["risk_class", "nearest_major_river"])
    test = df[df["event_time"] >= "2026-01-01"]
    print(f"Test set: {len(test):,} rows")
    print(test["nearest_major_river"].value_counts())

    X_test = test[features]
    y_test = test["risk_class"]

    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(model.predict(X_test))
    else:
        y_pred = model.predict(X_test)

    test = test.copy()
    test["y_pred"] = y_pred

    print("\n" + "="*70)
    print("PER-RIVER EVALUATION")
    print("="*70)

    rows = []
    for river in sorted(test["nearest_major_river"].dropna().unique()):
        river_mask = test["nearest_major_river"] == river
        river_test = test[river_mask]
        if len(river_test) < 10:
            print(f"\n{river}: only {len(river_test)} test rows -- too few for a reliable check")
            continue

        y_true_r = river_test["risk_class"]
        y_pred_r = river_test["y_pred"]

        acc = accuracy_score(y_true_r, y_pred_r)
        macro_f1 = f1_score(y_true_r, y_pred_r, average="macro")
        report = classification_report(y_true_r, y_pred_r, labels=LABELS, output_dict=True, zero_division=0)

        print(f"\n--- {river} (n={len(river_test):,}) ---")
        print(f"Accuracy: {acc:.3f}, Macro F1: {macro_f1:.3f}")
        for label in LABELS:
            print(f"  {label}: precision={report[label]['precision']:.3f}, "
                  f"recall={report[label]['recall']:.3f}, "
                  f"support={int(report[label]['support'])}")

        rows.append({
            "river": river, "n_test_rows": len(river_test),
            "accuracy": acc, "macro_f1": macro_f1,
            "danger_precision": report["Danger"]["precision"],
            "danger_recall": report["Danger"]["recall"],
            "danger_support": int(report["Danger"]["support"]),
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv("river_comparison_metrics.csv", index=False)

    print("\n" + "="*70)
    print("SUMMARY / INTERPRETATION")
    print("="*70)
    print(comparison.round(4).to_string(index=False))

    if len(comparison) == 2:
        f1_gap = abs(comparison["macro_f1"].iloc[0] - comparison["macro_f1"].iloc[1])
        danger_recall_gap = abs(comparison["danger_recall"].iloc[0] - comparison["danger_recall"].iloc[1])
        print(f"\nMacro F1 gap between rivers: {f1_gap:.3f}")
        print(f"Danger recall gap between rivers: {danger_recall_gap:.3f}")
        if f1_gap > 0.10 or danger_recall_gap > 0.15:
            print("\n*** MEANINGFUL DISPARITY -- the combined model performs notably")
            print("worse for one river than the other. Given the known Rapti data")
            print("sparsity, check which direction this gap runs (likely Rapti worse).")
            print("Worth documenting explicitly in your report as a known limitation,")
            print("and/or considering river-specific confidence flagging in the dashboard.")
        else:
            print("\nGap is modest -- the combined model appears reasonably river-fair")
            print("on this test set, despite the underlying data asymmetry. Still worth")
            print("stating this check was done in your report's validation section.")

    if "Danger" in LABELS:
        low_support_rivers = comparison[comparison["danger_support"] < 20]
        if len(low_support_rivers):
            print(f"\nCAUTION: {list(low_support_rivers['river'])} have fewer than 20 Danger")
            print("examples in the test set -- any precision/recall number for Danger on")
            print("that river individually is statistically noisy, not just possibly biased.")


if __name__ == "__main__":
    main()
