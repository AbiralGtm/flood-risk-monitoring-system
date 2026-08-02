"""
STEP 3 (FINAL, self-verifying) — Build the ML training dataset with the
CORRECTED multiplicative risk formula.

This version PRINTS A VERIFICATION CHECK at the end and will loudly
complain if the dry-day fix isn't behaving as expected, so there's no
ambiguity about whether the fix took effect.

FIX: STATIC_CSV corrected from "static_grid_with_stations.csv" (a typo
that never matched any file 00_assign_stations_to_grid.py actually
produces) to "static_grid_dataset_with_stations.csv", which is the real
output filename from that script. Without this fix, this script would
raise FileNotFoundError every time, regardless of anything else in the
pipeline.
"""

import pandas as pd
import numpy as np

STATIC_CSV = "static_grid_dataset_with_stations.csv"  # was: static_grid_with_stations.csv (typo/mismatch)
WEIGHTS_CSV = "grid_rainfall_idw_weights.csv"
RAINFALL_CSV = "rainfall_by_station_selected.csv"
RIVER_CSV = "river_level_by_station_selected.csv"
OUT_CSV = "ml_training_dataset.csv"

RAIN_COLS = ["rainfall_1h_mm", "rainfall_3h_mm", "rainfall_6h_mm",
             "rainfall_12h_mm", "rainfall_24h_mm", "rainfall_score"]

static = pd.read_csv(STATIC_CSV)
weights = pd.read_csv(WEIGHTS_CSV, dtype={"rainfall_station_id": str})
rainfall = pd.read_csv(RAINFALL_CSV, dtype={"rainfall_station_id": str}, parse_dates=["event_time"])

# ----------------------------------------------------------------------
# 1. Vectorized IDW blend
# ----------------------------------------------------------------------
stations = sorted(weights["rainfall_station_id"].unique())
s_idx = {s: i for i, s in enumerate(stations)}
grids = static["grid_id"].values
g_idx = {g: i for i, g in enumerate(grids)}

rainfall["rainfall_station_id"] = rainfall["rainfall_station_id"].astype(str)
unknown = set(rainfall["rainfall_station_id"].unique()) - set(stations)
if unknown:
    before_n = len(rainfall)
    rainfall = rainfall[rainfall["rainfall_station_id"].isin(stations)].copy()
    print(f"WARNING: dropped {before_n - len(rainfall)} rainfall rows for "
          f"station(s) {sorted(unknown)} not present in rainfall_stations.csv.")

W = np.zeros((len(grids), len(stations)))
for _, r in weights.iterrows():
    if r["grid_id"] in g_idx:
        W[g_idx[r["grid_id"]], s_idx[r["rainfall_station_id"]]] = r["weight"]

events = sorted(rainfall["event_id"].unique())
e_idx = {e: i for i, e in enumerate(events)}

avail = np.zeros((len(stations), len(events)), dtype=bool)
V = {c: np.zeros((len(stations), len(events))) for c in RAIN_COLS}
for _, r in rainfall.iterrows():
    si, ei = s_idx[r["rainfall_station_id"]], e_idx[r["event_id"]]
    avail[si, ei] = True
    for c in RAIN_COLS:
        V[c][si, ei] = r[c] if pd.notna(r[c]) else 0.0

print("Computing IDW-blended rainfall (vectorized)...")
Wsum_per_event = W @ avail.astype(float)
Wsum_per_event[Wsum_per_event == 0] = np.nan
blend = {}
for c in RAIN_COLS:
    raw = W @ (V[c] * avail)
    blend[c] = raw / Wsum_per_event

n_g, n_e = len(grids), len(events)
grid_rain = pd.DataFrame({
    "grid_id": np.repeat(grids, n_e),
    "event_id": np.tile(events, n_g),
})
for c in RAIN_COLS:
    grid_rain[c] = blend[c].ravel()
grid_rain = grid_rain.dropna(subset=["rainfall_score"])

ev_times = rainfall[["event_id", "event_time"]].drop_duplicates()
grid_rain = grid_rain.merge(ev_times, on="event_id")
print(f"Blended rainfall rows: {len(grid_rain):,}")

# ----------------------------------------------------------------------
# 2. Join to static attributes
# ----------------------------------------------------------------------
ml = static.merge(grid_rain, on="grid_id", how="inner")
print(f"After static join: {len(ml):,} rows")

# ----------------------------------------------------------------------
# 3. River level
# ----------------------------------------------------------------------
try:
    river = pd.read_csv(RIVER_CSV, parse_dates=["event_time"])
    riv_slim = river[["event_id", "river_station_id", "water_level_m", "river_level_score"]]
    ml = ml.merge(riv_slim, on=["event_id", "river_station_id"], how="left")
    matched = ml["river_level_score"].notna().sum()
    print(f"Rows with river-level match: {matched:,} / {len(ml):,} ({100*matched/len(ml):.0f}%)")

    # FIX: handle upstream-rain scenarios -- local rainfall can be near
    # zero while the river is already at/near danger level from rain
    # falling upstream (outside Bharatpur), with travel-time lag before
    # it arrives at the gauge. The plain 0.70/0.30 weighted average caps
    # dynamic_hazard_score at only 0.30 when rainfall_score=0, even if
    # river_level_score=1.0 (river AT danger level) -- underweighting a
    # genuinely dangerous situation. Fix: dynamic hazard is at least as
    # severe as the river's own reading, regardless of local rain.
    weighted_avg = 0.70 * ml["rainfall_score"] + 0.30 * ml["river_level_score"]
    river_floor = ml["river_level_score"].fillna(0)
    ml["dynamic_hazard_score"] = np.where(
        ml["river_level_score"].notna(),
        np.maximum(weighted_avg, river_floor),
        ml["rainfall_score"],
    )
except FileNotFoundError:
    print(f"NOTE: {RIVER_CSV} not found -- using rainfall-only dynamic hazard.")
    ml["dynamic_hazard_score"] = ml["rainfall_score"]

# ----------------------------------------------------------------------
# 4. FINAL RISK SCORE -- MULTIPLICATIVE FORMULA (THE FIX)
#
#    THIS IS THE LINE THAT MATTERS. If you are debugging why a fix
#    "isn't taking effect", check THIS EXACT LINE in THIS EXACT FILE
#    that Python is actually importing/running -- not a different copy
#    elsewhere on disk.
# ----------------------------------------------------------------------
ml["final_risk_score"] = ml["static_susceptibility_score"] * (0.2 + 0.8 * ml["dynamic_hazard_score"])

# Thresholds recalibrated for the NEW multiplicative score's distribution --
# the old fixed 0.25/0.50/0.75 cutoffs were designed for the additive
# formula's wider range and made Danger nearly impossible to reach (0.136%
# of all rows), risking zero Danger examples in any given time-based test
# split. These percentile-based breaks (70th/90th/98th of the actual
# final_risk_score distribution) keep Danger genuinely rare -- appropriate,
# since severe risk should be a minority -- while ensuring it's still
# representable in train/val/test splits.
breaks = ml["final_risk_score"].quantile([0.70, 0.90, 0.98]).values
print(f"Recalibrated risk class breaks (70th/90th/98th percentile): {breaks.round(4)}")

def classify_risk(s):
    if s <= breaks[0]: return "Low"
    if s <= breaks[1]: return "Medium"
    if s <= breaks[2]: return "High"
    return "Danger"

ml["risk_class"] = ml["final_risk_score"].apply(classify_risk)
ml["flood_label"] = ml["risk_class"].isin(["High", "Danger"]).astype(int)

if "nearest_major_river" in ml.columns:
    river_map = {r: i for i, r in enumerate(sorted(ml["nearest_major_river"].dropna().unique()))}
    ml["nearest_major_river_encoded"] = ml["nearest_major_river"].map(river_map)

for c in ["within_2km_narayani", "within_2km_rapti"]:
    if c in ml.columns and ml[c].dtype == object:
        ml[c] = ml[c].astype(str).str.upper().eq("TRUE").astype(int)

ml.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(ml):,} rows to {OUT_CSV}")
print("\nrisk_class distribution:")
print(ml["risk_class"].value_counts())
print(f"\nflood_label=1 fraction: {ml['flood_label'].mean():.3f}")

# ----------------------------------------------------------------------
# 5. SELF-VERIFICATION -- proves the fix actually ran, right here, right
#    now, on the file that was just written. No ambiguity.
# ----------------------------------------------------------------------
print("\n" + "="*70)
print("SELF-VERIFICATION CHECK")
print("="*70)
# Check GENUINELY calm conditions: dry rainfall AND normal river level.
# (Dry rainfall alone isn't sufficient -- a grid can legitimately show
# elevated risk if the river is already high from upstream causes even
# with no local rain. That's correct behavior, not a bug.)
if "river_level_score" in ml.columns:
    calm = ml[(ml["rainfall_score"] < 0.05) & (ml["river_level_score"].fillna(0) < 0.3)]
else:
    calm = ml[ml["rainfall_score"] < 0.05]
calm_hd_frac = calm["flood_label"].mean() if len(calm) else float("nan")
print(f"Genuinely-calm-row (dry rainfall AND normal river) High/Danger fraction: {calm_hd_frac:.1%}")
if calm_hd_frac < 0.02:
    print("PASS -- multiplicative formula confirmed working (should be ~0%).")
else:
    print("*** FAIL -- this should be near 0% for genuinely calm conditions. ***")

print(f"\nDanger-class representation check:")
for split_name, split_df in [("Train (<=2024)", ml[ml['event_time']<='2024-12-31']),
                               ("Val (2025)", ml[(ml['event_time']>='2025-01-01')&(ml['event_time']<='2025-12-31')]),
                               ("Test (2026)", ml[ml['event_time']>='2026-01-01'])]:
    n_danger = (split_df["risk_class"]=="Danger").sum()
    print(f"  {split_name}: {n_danger} Danger rows out of {len(split_df)}")
    if n_danger == 0:
        print(f"    *** WARNING: zero Danger examples in this split -- cannot evaluate Danger-class metrics here ***")
