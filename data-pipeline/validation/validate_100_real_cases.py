"""
Validate against 100+ REAL severe conditions pulled directly from your
test set (2026 data, never seen during training) -- not hand-picked
evidence points. This gives a statistically meaningful answer instead
of a 7-point anecdote.

Logic: pull rows where actual conditions were objectively severe by
domain criteria (river above warning level, OR top-5% rainfall) using
the RAW data, independent of what the model predicts. Then check how
often the model correctly flags these as High/Danger. Also checks 100
genuinely calm rows to confirm it doesn't cry wolf.

Run: python validate_100_real_cases.py
"""

import pandas as pd
import numpy as np
import joblib

DATASET_CSV = "ml_training_dataset.csv"
MODEL_PATH = "model_xgboost.joblib"

df = pd.read_csv(DATASET_CSV, parse_dates=["event_time"])
bundle = joblib.load(MODEL_PATH)
model, FEATURES = bundle["model"], bundle["features"]
label_encoder = bundle.get("label_encoder")

test = df[df["event_time"] >= "2026-01-01"].dropna(subset=FEATURES).copy()
print(f"Test set size (never seen in training): {len(test):,}")

# ----------------------------------------------------------------------
# Define "objectively severe" using RAW data, independent of the model:
# river above warning level (water_level_m >= warning_level_m if known),
# OR top-5% rainfall_24h_mm as a fallback for grids without river match.
# ----------------------------------------------------------------------
severe_river = test[test["water_level_m"] >= test.get("warning_level_m", 7.3)]
rain_threshold = test["rainfall_24h_mm"].quantile(0.95)
severe_rain = test[test["rainfall_24h_mm"] >= rain_threshold]
severe = pd.concat([severe_river, severe_rain]).drop_duplicates(subset=["grid_id", "event_id"])

print(f"Objectively severe rows found (river above warning OR top-5% rainfall): {len(severe):,}")

n_check = min(100, len(severe))
sample_severe = severe.sample(n=n_check, random_state=42)

pred_raw = model.predict(sample_severe[FEATURES])
preds = label_encoder.inverse_transform(pred_raw) if label_encoder is not None else pred_raw
correct = pd.Series(preds).isin(["High", "Danger"]).values

print(f"\n=== SEVERE CONDITIONS CHECK (n={n_check}) ===")
print(f"Model correctly flagged High/Danger: {correct.sum()} / {n_check} ({100*correct.mean():.1f}%)")
print("\nPrediction breakdown on severe rows:")
print(pd.Series(preds).value_counts())

# ----------------------------------------------------------------------
# Also check 100 genuinely calm rows -- confirms it doesn't over-warn
# ----------------------------------------------------------------------
calm = test[(test["rainfall_score"] < 0.05) & (test["river_level_score"].fillna(0) < 0.3)]
n_calm = min(100, len(calm))
sample_calm = calm.sample(n=n_calm, random_state=42)
pred_calm_raw = model.predict(sample_calm[FEATURES])
preds_calm = label_encoder.inverse_transform(pred_calm_raw) if label_encoder is not None else pred_calm_raw
false_alarms = pd.Series(preds_calm).isin(["High", "Danger"]).sum()

print(f"\n=== CALM CONDITIONS CHECK (n={n_calm}) ===")
print(f"Model incorrectly flagged High/Danger on calm days: {false_alarms} / {n_calm} "
      f"({100*false_alarms/n_calm:.1f}%) -- should be near 0%")

print("\n=== SUMMARY ===")
print(f"On {n_check} genuinely severe real conditions: {100*correct.mean():.1f}% correctly caught")
print(f"On {n_calm} genuinely calm real conditions: {100*(1-false_alarms/n_calm):.1f}% correctly stayed quiet")
