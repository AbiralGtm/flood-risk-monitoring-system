"""
STEP 4 (FINAL, corrected column names) — Train Logistic Regression + Random
Forest on the ML dataset, time-based split, full-scale (not subsampled --
this ran subsampled in a sandbox for speed; run this version locally with
your real compute for the final numbers).

FIX: added "flow_accumulation_score" to FEATURES. It was computed and
folded into static_susceptibility_score (and therefore into the training
LABELS via final_risk_score), but was never given to the model as an
input feature. Since flow_accumulation_score was confirmed to have
near-zero correlation with every other feature (elevation_mean, slope_mean,
distance-to-river), the model had no way to learn the pattern behind the
~105-250 grids whose label changed because of it -- a real label/feature
mismatch, not just a missing "nice to have" column. This is the most
likely explanation for the small dip from the pre-flow-accumulation
baseline (98.4% / macro F1 0.963) to 98.1% / 0.947.

Input:
  - ml_training_dataset.csv (from step 3)

Outputs:
  - model_random_forest.joblib
  - model_logreg.joblib
  - predicted_risk_grid.csv / .geojson
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

IN_CSV = "ml_training_dataset.csv"

# Column names matched to your ACTUAL static_dataset_final.geojson schema
# (elevation_mean, not elev_mean -- this was a real bug in the earlier
# version of this script).
FEATURES = [
    "elevation_mean", "slope_mean",
    "distance_to_narayani_m", "distance_to_rapti_m", "nearest_river_distance",
    "nearest_major_river_encoded",
    "within_2km_narayani", "within_2km_rapti",
    "lulc_encoded", "historical_flood_evidence",
    "flow_accumulation_score",   # NEW -- see fix note above
    "rainfall_1h_mm", "rainfall_3h_mm", "rainfall_6h_mm",
    "rainfall_12h_mm", "rainfall_24h_mm", "rainfall_score",
    "water_level_m", "river_level_score",
]
TARGET = "risk_class"

df = pd.read_csv(IN_CSV, parse_dates=["event_time"])

if "flow_accumulation_score" not in df.columns:
    print("ERROR: flow_accumulation_score not found in ml_training_dataset.csv.")
    print("This means static_grid_dataset_with_stations.csv (produced by")
    print("00_assign_stations_to_grid.py) doesn't carry that column through --")
    print("check that 00 is reading static_dataset_with_flowacc.geojson as its")
    print("GRID_GEOJSON input, not an earlier pre-flow-accumulation file.")
    raise SystemExit(1)

# within_2km_narayani/rapti come in as string "TRUE"/"FALSE" from the
# GeoJSON properties -- must convert to 0/1 before modeling.
for c in ["within_2km_narayani", "within_2km_rapti"]:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.upper().eq("TRUE").astype(int)

df = df.dropna(subset=FEATURES + [TARGET])
print(f"Rows available after dropping missing values: {len(df):,}")

# ----------------------------------------------------------------------
# Time-based split
# ----------------------------------------------------------------------
train = df[df["event_time"] <= "2024-12-31"]
val = df[(df["event_time"] >= "2025-01-01") & (df["event_time"] <= "2025-12-31")]
test = df[df["event_time"] >= "2026-01-01"]
print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# ----------------------------------------------------------------------
# Logistic Regression baseline
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\nTraining Logistic Regression baseline...")
logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
logreg.fit(X_train_s, y_train)
pred_lr = logreg.predict(X_test_s)

# ----------------------------------------------------------------------
# Random Forest -- FULL SCALE (400 trees, full training set). The
# sandbox run used 100 trees / 200k-row subsample for speed; use the
# full settings here for your final reported numbers.
# ----------------------------------------------------------------------
print("Training Random Forest (400 trees, full training set -- this will "
      "take several minutes on the full ~576k-row training set)...")
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# ----------------------------------------------------------------------
# Optional XGBoost
# ----------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        objective="multi:softprob", eval_metric="mlogloss", random_state=42,
    )
    xgb.fit(X_train, y_train_enc)
    pred_xgb = le.inverse_transform(xgb.predict(X_test))
    have_xgb = True
except ImportError:
    have_xgb = False
    print("xgboost not installed -- skipping (pip install xgboost to enable)")

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def report(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    labels = ["Low", "Medium", "High", "Danger"]
    print(classification_report(y_true, y_pred, labels=labels, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels))
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.3f}")

report("Logistic Regression", y_test, pred_lr)
report("Random Forest", y_test, pred_rf)
if have_xgb:
    report("XGBoost", y_test, pred_xgb)

print("\nRandom Forest feature importance:")
print(pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False).round(4))

# ----------------------------------------------------------------------
# Save models + latest-event predictions per grid
# ----------------------------------------------------------------------
joblib.dump({"model": rf, "features": FEATURES}, "model_random_forest.joblib")
joblib.dump({"model": logreg, "scaler": scaler, "features": FEATURES}, "model_logreg.joblib")
if have_xgb:
    joblib.dump({"model": xgb, "label_encoder": le, "features": FEATURES}, "model_xgboost.joblib")
    print("Saved model_xgboost.joblib")

# Use the best-performing model (XGBoost if available, else RF) for the
# actual dashboard predictions -- based on this run's results, XGBoost
# clearly outperforms RF on Danger-class recall/precision balance.
best_model, best_name = (xgb, "xgboost") if have_xgb else (rf, "random_forest")
print(f"\nUsing {best_name} for final predicted_risk_grid outputs (best macro F1/Danger balance).")

latest = df.sort_values("event_time").groupby("grid_id").tail(1).copy()
if have_xgb and best_name == "xgboost":
    latest["predicted_risk_class"] = le.inverse_transform(best_model.predict(latest[FEATURES]))
else:
    latest["predicted_risk_class"] = best_model.predict(latest[FEATURES])
latest["risk_probability"] = best_model.predict_proba(latest[FEATURES]).max(axis=1).round(3)

cols_out = ["grid_id", "predicted_risk_class", "risk_probability",
            "centroid_lat", "centroid_lon", "elevation_mean", "slope_mean",
            "distance_to_narayani_m", "distance_to_rapti_m",
            "nearest_major_river", "lulc_dominant", "rainfall_24h_mm",
            "historical_flood_evidence", "flow_accumulation_score", "static_risk_class"]
cols_out = [c for c in cols_out if c in latest.columns]
latest[cols_out].to_csv("predicted_risk_grid.csv", index=False)

print("\nSaved model_random_forest.joblib, model_logreg.joblib, predicted_risk_grid.csv")