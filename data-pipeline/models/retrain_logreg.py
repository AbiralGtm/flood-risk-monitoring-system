"""
retrain_logreg_pipeline.py

Retrains Logistic Regression as a single, self-contained sklearn Pipeline
(StandardScaler + LogisticRegression), fixing the root cause of the
0.015 accuracy bug: the previous model_logreg.joblib was trained on
scaled features, but the scaler itself was never saved, so downstream
scripts fed it raw, unscaled data at prediction time.

With a Pipeline, scaling happens automatically inside model.predict() /
model.predict_proba() -- no other script needs to change, and this class
of bug becomes structurally impossible going forward.

Matches the same conventions established for the other models in this
project: time-based split, river_level_score fallback fill before
dropna, and source-reliability sample weighting.

INPUT:  ml_training_dataset_v2.csv
OUTPUT: model_logreg.joblib  (overwrite the old, broken one)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

FEATURES = [
    "elevation_mean", "slope_mean", "distance_to_narayani_m", "distance_to_rapti_m",
    "nearest_river_distance", "nearest_major_river_encoded", "within_2km_narayani",
    "within_2km_rapti", "lulc_encoded", "historical_flood_evidence",
    "flow_accumulation_score", "rainfall_24h_mm", "rainfall_score", "river_level_score",
]
LABELS = ["Low", "Medium", "High", "Danger"]


def main():
    print("Loading dataset...")
    df = pd.read_csv("ml_training_dataset_v2.csv", parse_dates=["event_time"], low_memory=False)

    for c in ["within_2km_narayani", "within_2km_rapti"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.upper().eq("TRUE").astype(int)

    n_river_nan = df["river_level_score"].isna().sum() if "river_level_score" in df.columns else 0
    if n_river_nan > 0:
        print(f"Filling {n_river_nan:,} NaN river_level_score values with 0.3 "
              f"(established fallback convention) before dropna.")
        df["river_level_score"] = df["river_level_score"].fillna(0.3)

    df = df.dropna(subset=FEATURES + ["risk_class"])

    train = df[df["event_time"] <= "2025-12-31"]
    test = df[df["event_time"] >= "2026-01-01"]
    print(f"Train: {len(train):,} rows, Test: {len(test):,} rows")

    sample_weight = None
    if "data_source" in train.columns:
        sample_weight = np.where(train["data_source"] == "reanalysis_daily", 0.5, 1.0)
        print("Applying sample weights: observed_hourly=1.0, reanalysis_daily=0.5")

    print("\nTraining Logistic Regression as a StandardScaler + LogisticRegression Pipeline...")
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    fit_kwargs = {"logisticregression__sample_weight": sample_weight} if sample_weight is not None else {}
    pipeline.fit(train[FEATURES], train["risk_class"], **fit_kwargs)

    print("\nEvaluating on held-out 2026 test set...")
    y_pred = pipeline.predict(test[FEATURES])
    report = classification_report(test["risk_class"], y_pred, labels=LABELS, zero_division=0)
    print(report)

    pred_counts = pd.Series(y_pred).value_counts()
    print("Predicted class distribution (sanity check -- should NOT collapse to one class):")
    print(pred_counts)
    if pred_counts.iloc[0] / len(y_pred) > 0.90:
        print("\n*** WARNING: predictions are heavily collapsed toward one class. ***")
        print("*** This may indicate the scaling fix did not fully resolve the issue. ***")
    else:
        print("\nPrediction distribution looks reasonable -- no collapse detected.")

    joblib.dump({"model": pipeline, "features": FEATURES}, "model_logreg.joblib")
    print("\nSaved: model_logreg.joblib (Pipeline: StandardScaler + LogisticRegression)")
    print("No changes needed to generate_report_graphs.py -- predict()/predict_proba()")
    print("on a Pipeline scale internally, exactly like a plain model would.")


if __name__ == "__main__":
    main()