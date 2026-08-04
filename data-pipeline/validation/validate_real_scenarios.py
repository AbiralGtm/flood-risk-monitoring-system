"""
Real-scenario validation: check if your trained model predicts High/Danger
at ACTUAL documented flood locations, on their ACTUAL dates. Uses your
BIPAD-verified evidence points (real coordinates, real dates, verified=true)
instead of satellite imagery -- no revisit-cycle timing problems.

Run: python validate_real_scenarios.py
"""

import pandas as pd
import numpy as np
import joblib

EVIDENCE_CSV = "flood_evidence_bharatpur.csv"
DATASET_CSV = "ml_training_dataset.csv"
MODEL_PATH = "model_xgboost.joblib"

# Only records with a specific, real date (skip vague/placeholder dates)
DATED_EVIDENCE_IDS = ["EV001", "EV002", "EV003", "EV011", "EV012", "EV013",
                       "EV016", "EV019", "EV022"]

evidence = pd.read_csv(EVIDENCE_CSV)
evidence = evidence[evidence["evidence_id"].isin(DATED_EVIDENCE_IDS)]

df = pd.read_csv(DATASET_CSV, parse_dates=["event_time"])
bundle = joblib.load(MODEL_PATH)
model, FEATURES = bundle["model"], bundle["features"]
label_encoder = bundle.get("label_encoder")

# Grid centroids for nearest-grid lookup
grids = df.drop_duplicates(subset="grid_id")[["grid_id", "centroid_lat", "centroid_lon"]]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

unique_times = df["event_time"].drop_duplicates().sort_values().reset_index(drop=True)

results = []
for _, ev in evidence.iterrows():
    # Find nearest grid
    dists = haversine_km(ev["latitude"], ev["longitude"], grids["centroid_lat"], grids["centroid_lon"])
    nearest_idx = dists.idxmin()
    nearest_grid = grids.loc[nearest_idx, "grid_id"]
    nearest_dist_km = dists.loc[nearest_idx]

    # Find the WORST event within +/-2 days for THIS specific grid --
    # same class of bug as the earlier 02_rainfall_event_selection.py fix:
    # picking "nearest to midnight" grabs whichever hour is arithmetically
    # closest to 00:00, not the actual worst conditions. Search a window
    # and pick the event with the highest final_risk_score for this grid.
    target = pd.Timestamp(ev["event_date"], tz="UTC")
    window = df[(df["grid_id"] == nearest_grid) &
                (df["event_time"] >= target - pd.Timedelta(days=2)) &
                (df["event_time"] <= target + pd.Timedelta(days=2))]
    window = window.dropna(subset=FEATURES + ["final_risk_score"])
    if len(window) == 0:
        results.append({"evidence_id": ev["evidence_id"], "location": ev["location_name"],
                         "date": ev["event_date"], "result": "NO DATA for this grid/event"})
        continue
    worst_idx = window["final_risk_score"].idxmax()
    row = window.loc[[worst_idx]]
    closest_time = row["event_time"].iloc[0]
    days_off = abs((closest_time - target).days)

    pred_raw = model.predict(row[FEATURES])[0]
    pred = label_encoder.inverse_transform([pred_raw])[0] if label_encoder is not None else pred_raw
    confidence = model.predict_proba(row[FEATURES]).max()

    results.append({
        "evidence_id": ev["evidence_id"],
        "location": ev["location_name"],
        "confidence_level": ev["confidence_level"],
        "date": ev["event_date"],
        "nearest_grid": nearest_grid,
        "grid_distance_km": round(nearest_dist_km, 2),
        "closest_event_used": closest_time,
        "days_off_from_real_date": days_off,
        "model_predicted": pred,
        "model_confidence": round(confidence, 3),
        "correct": pred in ["High", "Danger"],
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

if "correct" in results_df.columns:
    valid = results_df[results_df["correct"].notna()]
    if len(valid):
        print(f"\nCorrectly flagged High/Danger: {valid['correct'].sum()} / {len(valid)} "
              f"({100*valid['correct'].mean():.0f}%)")
        print("\nCheck 'grid_distance_km' and 'days_off_from_real_date' columns --")
        print("large values mean the match is less reliable (evidence far from any")
        print("grid, or no event close to the real date).")

results_df.to_csv("real_scenario_validation.csv", index=False)
print("\nSaved real_scenario_validation.csv")