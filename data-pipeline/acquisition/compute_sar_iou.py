"""
Compare SAR-observed flood extent (Sept 28, 2024) against your trained
XGBoost model's predicted High/Danger grids for the closest actual event
to that date. This is the actual validation payoff -- an independent
check against satellite observation, not just self-consistency with your
own weak labels.

Inputs:
  - sar_flooded_grids_2024-09-28.csv   (from sar_intersect_grid.py)
  - ml_training_dataset.csv             (has grid_id, event_time, static features)
  - model_xgboost.joblib

Output: printed IoU, precision, recall, and a CSV of per-grid agreement
for visual mapping in QGIS if you want to see it spatially.
"""

import pandas as pd
import joblib

SAR_CSV = "sar_flooded_grids_2024-09-28.csv"
DATASET_CSV = "ml_training_dataset.csv"
MODEL_PATH = "model_xgboost.joblib"
OUT_CSV = "sar_vs_model_comparison.csv"

# ----------------------------------------------------------------------
# 1. Load SAR ground truth
# ----------------------------------------------------------------------
sar = pd.read_csv(SAR_CSV)
sar_flooded_ids = set(sar["grid_id"])
print(f"SAR-observed flooded grids: {len(sar_flooded_ids)}")

# ----------------------------------------------------------------------
# 2. Find the actual event closest to Sept 28, 2024 in your dataset
# ----------------------------------------------------------------------
df = pd.read_csv(DATASET_CSV, parse_dates=["event_time"])
bundle = joblib.load(MODEL_PATH)
model, FEATURES = bundle["model"], bundle["features"]
label_encoder = bundle.get("label_encoder")

target = pd.Timestamp("2024-09-28", tz="UTC")
unique_times = df["event_time"].drop_duplicates().sort_values().reset_index(drop=True)
closest_time = unique_times.iloc[(unique_times - target).abs().idxmin()]
print(f"Closest actual event in dataset: {closest_time}")

event_rows = df[df["event_time"] == closest_time].dropna(subset=FEATURES).copy()
print(f"Grids available for this event: {len(event_rows)}")

# ----------------------------------------------------------------------
# 3. Get model predictions for this event
# ----------------------------------------------------------------------
pred_raw = model.predict(event_rows[FEATURES])
event_rows["predicted_risk_class"] = (
    label_encoder.inverse_transform(pred_raw) if label_encoder is not None else pred_raw
)
event_rows["predicted_flood"] = event_rows["predicted_risk_class"].isin(["High", "Danger"])

predicted_flooded_ids = set(event_rows[event_rows["predicted_flood"]]["grid_id"])
print(f"Model-predicted High/Danger grids: {len(predicted_flooded_ids)}")

# ----------------------------------------------------------------------
# 4. Compute overlap metrics
# ----------------------------------------------------------------------
intersection = sar_flooded_ids & predicted_flooded_ids
union = sar_flooded_ids | predicted_flooded_ids

iou = len(intersection) / len(union) if union else float("nan")
precision = len(intersection) / len(predicted_flooded_ids) if predicted_flooded_ids else float("nan")
recall = len(intersection) / len(sar_flooded_ids) if sar_flooded_ids else float("nan")

print("\n" + "="*60)
print("SAR vs MODEL VALIDATION RESULTS")
print("="*60)
print(f"Intersection (agree, both say flooded): {len(intersection)}")
print(f"Union (either says flooded):            {len(union)}")
print(f"IoU (Intersection over Union):          {iou:.3f}")
print(f"Precision (of model's flood calls, how many SAR confirmed): {precision:.3f}")
print(f"Recall (of SAR-observed floods, how many model caught):     {recall:.3f}")

print("\nInterpretation:")
print("IoU > 0.5 is generally considered good spatial agreement in flood")
print("mapping literature. Remember known SAR limitations: it under-detects")
print("flooding under tree cover and in built-up areas, so recall may look")
print("artificially low if many SAR-missed grids are forested/urban --")
print("check the land cover of any disagreement grids before concluding")
print("the MODEL is wrong.")

# ----------------------------------------------------------------------
# 5. Save per-grid comparison for spatial inspection in QGIS
# ----------------------------------------------------------------------
all_ids = sar_flooded_ids | predicted_flooded_ids
comparison = pd.DataFrame({"grid_id": list(all_ids)})
comparison["sar_observed_flood"] = comparison["grid_id"].isin(sar_flooded_ids)
comparison["model_predicted_flood"] = comparison["grid_id"].isin(predicted_flooded_ids)
comparison["agreement"] = comparison.apply(
    lambda r: "Both agree" if r["sar_observed_flood"] and r["model_predicted_flood"]
    else "SAR only (model missed)" if r["sar_observed_flood"]
    else "Model only (false positive or SAR missed it)", axis=1
)
lulc_col = ["grid_id", "lulc_dominant"] if "lulc_dominant" in event_rows.columns else ["grid_id"]
comparison = comparison.merge(event_rows[lulc_col], on="grid_id", how="left")
comparison.to_csv(OUT_CSV, index=False)
print(f"\nSaved per-grid comparison to {OUT_CSV} -- join this back to your")
print("grid geometry in QGIS to visualize where the model and SAR agree/disagree.")

if "lulc_dominant" in comparison.columns:
    print("\nDisagreement breakdown by land cover (checking for tree-cover/urban bias):")
    disagree = comparison[comparison["agreement"] != "Both agree"]
    print(disagree.groupby(["agreement", "lulc_dominant"]).size())
