"""
STEP 4 (FINAL, consistent with ml_training_dataset_v2.csv) — Train
Logistic Regression, Random Forest, and XGBoost on the fully-integrated
training set (observed 2021-2026 hourly BIPAD data + reconstructed
1993-2020 CHIRPS/GloFAS historical data), time-based split.

CHANGES FROM THE OBSERVED-ONLY VERSION
-----------------------------------------
1. IN_CSV now points to ml_training_dataset_v2.csv (the output of
   10_merge_and_finalize_training_data.py), not the old observed-only
   ml_training_dataset.csv.

2. FEATURES excludes rainfall_1h_mm, rainfall_3h_mm, rainfall_6h_mm,
   rainfall_12h_mm. CONFIRMED via direct inspection of the real merged
   file: these columns are EXACTLY 0.0 (not NaN) for every historical
   (reanalysis_daily) row, because CHIRPS only provides a daily value
   and 09's IDW blend treats missing sub-daily readings as literal zero
   rainfall for blending purposes. Leaving them in FEATURES would mean
   training the model to believe "zero rain in the last hour" on your
   worst historical flood days (180+mm/24h) -- a real, silent data
   corruption, not just a missing-data inconvenience.
   rainfall_24h_mm and rainfall_score ARE genuine, real values for every
   row regardless of era, and now carry the rainfall signal alone.

3. data_source is NOT included in FEATURES. Feeding it directly as an
   input would trivially let the model key off it, defeating the point
   of checking whether the model organically leans on data provenance.
   Instead, this script trains WITHOUT that column, then checks
   performance split by data_source AFTER training -- if accuracy/recall
   differ sharply between observed_hourly and reanalysis_daily rows in
   the test set, that's the signal (per the original research report)
   to consider splitting into two model tiers instead of one merged
   model. data_source is kept in the dataframe for this post-hoc check
   only, never passed to the model itself.

Input:
  - ml_training_dataset_v2.csv (from 10_merge_and_finalize_training_data.py)

Outputs:
  - model_random_forest.joblib
  - model_logreg.joblib
  - model_xgboost.joblib (if xgboost installed)
  - predicted_risk_grid.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

IN_CSV = "ml_training_dataset_v2.csv"

FEATURES = [
    "elevation_mean", "slope_mean",
    "distance_to_narayani_m", "distance_to_rapti_m", "nearest_river_distance",
    "nearest_major_river_encoded",
    "within_2km_narayani", "within_2km_rapti",
    "lulc_encoded", "historical_flood_evidence",
    "flow_accumulation_score",
    "rainfall_24h_mm", "rainfall_score",
    # rainfall_1h_mm/3h_mm/6h_mm/12h_mm deliberately EXCLUDED -- see
    # docstring above. Do not add these back without re-verifying they
    # aren't literal 0.0 for every historical row.
    "river_level_score",
    # water_level_m deliberately EXCLUDED -- confirmed via real-data check:
    # NaN for 983,762 rows (38% of the full dataset), overwhelmingly
    # RV02/Rapti grids (920,160 rows) plus a smaller genuine RV01/Narayani
    # gap (63,602 rows). This affects BOTH observed_hourly (188,562 rows,
    # 24% of real BIPAD data) and reanalysis_daily rows -- a real-world
    # telemetry gap, not just a historical-reconstruction artifact. Unlike
    # river_level_score (which has an established 0.3 "unknown condition"
    # fallback used consistently elsewhere in this pipeline), water_level_m
    # has no safe fallback value, and it's largely redundant with
    # river_level_score anyway -- dropping it lets us keep these rows via
    # the fillna below instead of losing over a third of the dataset.
]

TARGET = "risk_class"

df = pd.read_csv(IN_CSV, parse_dates=["event_time"], low_memory=False)
print(f"Loaded {len(df):,} rows from {IN_CSV}")

if "flow_accumulation_score" not in df.columns:
    print("ERROR: flow_accumulation_score not found -- check your pipeline produced")
    print("this file from the flow-accumulation-integrated static data.")
    raise SystemExit(1)

if "data_source" in df.columns:
    print(df["data_source"].value_counts())
else:
    print("WARNING: no data_source column found -- the post-training provenance")
    print("check at the end of this script will be skipped.")

# Defensive check: confirm the false-zero issue really is what we think it is,
# and warn loudly if rainfall_24h_mm/rainfall_score themselves look suspect too.
if "data_source" in df.columns:
    hist_mask = df["data_source"] == "reanalysis_daily"
    if hist_mask.any():
        pct_zero_24h = (df.loc[hist_mask, "rainfall_24h_mm"] == 0).mean() * 100
        if pct_zero_24h > 50:
            print(f"WARNING: {pct_zero_24h:.0f}% of historical rainfall_24h_mm values are "
                  f"exactly 0 -- this column may ALSO be corrupted, investigate before trusting "
                  f"model results.")

# within_2km_narayani/rapti come in as string "TRUE"/"FALSE" from the
# GeoJSON properties -- must convert to 0/1 before modeling.
for c in ["within_2km_narayani", "within_2km_rapti"]:
    if c in df.columns and df[c].dtype == object:
        df[c] = df[c].astype(str).str.upper().eq("TRUE").astype(int)

n_river_nan_before = df["river_level_score"].isna().sum()
if n_river_nan_before > 0:
    print(f"\nFilling {n_river_nan_before:,} NaN river_level_score values with 0.3 "
          f"(the 'unknown river condition' fallback already used consistently "
          f"elsewhere in this pipeline -- see 03/06/09), instead of dropping "
          f"these rows entirely. Confirmed via real-data check: mostly RV02/Rapti "
          f"grids (a genuine telemetry gap affecting both observed and historical "
          f"eras), plus a smaller RV01/Narayani gap.")
    df["river_level_score"] = df["river_level_score"].fillna(0.3)

before_n = len(df)
df = df.dropna(subset=FEATURES + [TARGET])
print(f"Rows available after dropping missing values: {len(df):,} "
      f"(dropped {before_n - len(df):,})")
if "data_source" in df.columns:
    print("Remaining rows by data_source after dropna:")
    print(df["data_source"].value_counts())

# ----------------------------------------------------------------------
# Time-based split
# ----------------------------------------------------------------------
train = df[df["event_time"] <= "2024-12-31"]
val = df[(df["event_time"] >= "2025-01-01") & (df["event_time"] <= "2025-12-31")]
test = df[df["event_time"] >= "2026-01-01"]
print(f"\nTrain: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

X_train, y_train = train[FEATURES], train[TARGET]
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
# SOURCE-RELIABILITY SAMPLE WEIGHTS
#
# Real finding from an earlier training run: merging historical data
# WITHOUT weighting dropped XGBoost's Danger-class precision from 0.892
# to 0.518 (recall stayed roughly flat, 0.916 -> 0.927) -- the classic
# signature of a model becoming more trigger-happy about calling Danger,
# not more accurate. Since the test set is 100% clean 2026 observed data
# (historical rows can never reach it -- see note below), this is a real
# training-time effect, not test contamination.
#
# Root cause: the training set is ~73% historical rows once time-split
# (historical falls entirely in train), and roughly half of historical
# river_level_score values are the 0.3 "unknown" fallback rather than a
# real reading, plus evidence-override rows carry deliberately blunt
# floor values (0.85-0.95) rather than naturally-varying signal. Sheer
# volume of this coarser signal was outweighing the more precisely-
# labeled real hourly data.
#
# Fix: downweight reanalysis_daily rows during training so the model
# still gains COVERAGE of known catastrophic years without letting
# VOLUME dominate the decision boundary. HISTORICAL_WEIGHT=0.5 is a
# reasonable starting point (half-trust vs. observed), not a precisely
# derived optimum -- if you have time, sweep a couple of values (e.g.
# 0.3, 0.5, 0.7) and pick whichever gives the best Danger-class
# precision/recall balance on the 2025 validation split (val is real
# observed data too, so it's a fair tuning target).
# ----------------------------------------------------------------------
HISTORICAL_WEIGHT = 0.5
if "data_source" in train.columns:
    sample_weight = np.where(train["data_source"] == "reanalysis_daily", HISTORICAL_WEIGHT, 1.0)
    print(f"\nApplying sample weights: observed_hourly=1.0, "
          f"reanalysis_daily={HISTORICAL_WEIGHT} (source-reliability downweighting)")
else:
    sample_weight = None
    print("\nWARNING: no data_source column -- training without source-reliability weights.")

# ----------------------------------------------------------------------
# Random Forest -- FULL SCALE
# ----------------------------------------------------------------------
print(f"Training Random Forest (400 trees, {len(X_train):,}-row training set -- "
      f"this may take a while)...")
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train, sample_weight=sample_weight)
pred_rf = rf.predict(X_test)

# ----------------------------------------------------------------------
# XGBoost
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
    xgb.fit(X_train, y_train_enc, sample_weight=sample_weight)
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
# POST-HOC data provenance check -- data_source was NEVER given to the
# model as a feature.
#
# IMPORTANT: because historical (reconstructed) data only spans
# 1993-2020 and the test split requires event_time >= 2026, historical
# rows can NEVER appear in the test set -- they fall entirely within
# "train" by construction. This is actually good news for your reported
# test metrics (they're evaluating purely on real observed 2026 data,
# uncontaminated by reconstruction), but it means we can't compare
# test-set performance by data_source -- there's only one source there.
#
# Instead, this checks IN-SAMPLE prediction behavior on the training set
# split by data_source. This is NOT a generalization test (the model has
# seen these rows), but it can still reveal gross behavioral differences
# -- e.g. if the model fits historical rows dramatically worse than
# observed rows even in-sample, that's a sign the reconstructed features
# don't relate to the labels as cleanly as the real hourly data does.
# ----------------------------------------------------------------------
if "data_source" in train.columns:
    print("\n" + "="*70)
    print("DATA PROVENANCE CHECK (in-sample, on TRAINING data)")
    print("="*70)
    print("NOTE: historical (reanalysis_daily) rows only span 1993-2020 and")
    print("therefore fall ENTIRELY within the training set -- none reach the")
    print("2026+ test split. Your test metrics above are already a clean,")
    print("uncontaminated evaluation on real observed data only. This check")
    print("is in-sample (not a generalization test) and only screens for")
    print("gross behavioral differences between sources.\n")

    best_model_obj = xgb if have_xgb else rf
    if have_xgb:
        train_pred = le.inverse_transform(best_model_obj.predict(X_train))
    else:
        train_pred = best_model_obj.predict(X_train)

    train_check = train.copy()
    train_check["pred"] = train_pred
    train_check["correct"] = (train_check["pred"] == train_check[TARGET]).astype(int)
    print(train_check.groupby("data_source")["correct"].agg(["mean", "count"]))
    print("\nIf in-sample fit is dramatically worse for reanalysis_daily rows")
    print("(e.g. >15-20 percentage points below observed_hourly), that suggests")
    print("the reconstructed features are noisier or less predictive of the")
    print("labels than your real hourly data -- worth investigating further")
    print("before trusting the historical data's contribution.")

# ----------------------------------------------------------------------
# Save models + latest-event predictions per grid
# ----------------------------------------------------------------------
joblib.dump({"model": rf, "features": FEATURES}, "model_random_forest.joblib")
joblib.dump({"model": logreg, "scaler": scaler, "features": FEATURES}, "model_logreg.joblib")
if have_xgb:
    joblib.dump({"model": xgb, "label_encoder": le, "features": FEATURES}, "model_xgboost.joblib")
    print("\nSaved model_xgboost.joblib")

best_model, best_name = (xgb, "xgboost") if have_xgb else (rf, "random_forest")
print(f"Using {best_name} for final predicted_risk_grid outputs.")

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