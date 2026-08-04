"""
Checks whether the model's predictions on "severe weather" rows actually
DISAGREE with the ground truth (risk_class from your label formula), or
whether they correctly AGREE that severe weather + safe location = still
Medium, not High/Danger. This distinguishes real model error from correct,
location-aware behavior.

Run: python check_severe_vs_groundtruth.py
"""

import pandas as pd
import joblib

DATASET_CSV = "ml_training_dataset.csv"
MODEL_PATH = "model_xgboost.joblib"

df = pd.read_csv(DATASET_CSV, parse_dates=["event_time"])
bundle = joblib.load(MODEL_PATH)
model, FEATURES = bundle["model"], bundle["features"]
label_encoder = bundle.get("label_encoder")

test = df[df["event_time"] >= "2026-01-01"].dropna(subset=FEATURES + ["risk_class"]).copy()

severe_river = test[test["water_level_m"] >= test.get("warning_level_m", 7.3)]
rain_threshold = test["rainfall_24h_mm"].quantile(0.95)
severe_rain = test[test["rainfall_24h_mm"] >= rain_threshold]
severe = pd.concat([severe_river, severe_rain]).drop_duplicates(subset=["grid_id", "event_id"])

sample = severe.sample(n=min(100, len(severe)), random_state=42)

pred_raw = model.predict(sample[FEATURES])
sample["model_predicted"] = label_encoder.inverse_transform(pred_raw) if label_encoder is not None else pred_raw

# THE KEY CHECK: does the model's prediction match the ACTUAL ground
# truth label for these rows? This is different from "is it High/Danger" --
# it directly checks agreement with what the row was actually labeled.
sample["matches_ground_truth"] = sample["model_predicted"] == sample["risk_class"]

print(f"n = {len(sample)}")
print(f"\nModel exactly matches ground truth risk_class: "
      f"{sample['matches_ground_truth'].sum()} / {len(sample)} "
      f"({100*sample['matches_ground_truth'].mean():.1f}%)")

print("\nGround truth risk_class distribution on these 'severe weather' rows:")
print(sample["risk_class"].value_counts())
print("\n^ If many of these are actually 'Medium' in the ground truth (not")
print("High/Danger), that confirms severe weather alone doesn't guarantee")
print("high risk everywhere -- location matters too, and the model agreeing")
print("with 'Medium' ground truth is CORRECT behavior, not an error.")

print("\nCross-tab: true risk_class (rows) vs model prediction (columns):")
print(pd.crosstab(sample["risk_class"], sample["model_predicted"]))

# How far off, when they DO disagree?
order = {"Low": 0, "Medium": 1, "High": 2, "Danger": 3}
sample["true_rank"] = sample["risk_class"].map(order)
sample["pred_rank"] = sample["model_predicted"].map(order)
sample["rank_diff"] = (sample["true_rank"] - sample["pred_rank"]).abs()
print(f"\nOf the disagreements, how many are just 1 class off (adjacent) vs "
      f"2+ classes off (real errors)?")
disagreements = sample[~sample["matches_ground_truth"]]
if len(disagreements):
    print(disagreements["rank_diff"].value_counts().sort_index())
