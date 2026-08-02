"""
STEP 2 (FINAL) — Select representative event-times from your real BIPAD
rainfall data, force-include flood incident dates, prepare per-station
rainfall and river-level tables for those events.

Fixes applied after the real run: BIPAD's timestamps have mixed formats
across records (some with microseconds, some without, mixed timezone
representations) -- use format='mixed' and floor to the hour so different
stations' slightly-offset reading times align to the same event_time.

Inputs:
  - rainfall_hourly_multistation.csv  (cleaned, deduplicated stations)
  - river_level_hourly_multistation.csv
  - flood_evidence_bharatpur.csv

Outputs:
  - selected_events.csv
  - rainfall_by_station_selected.csv
  - river_level_by_station_selected.csv
"""

import pandas as pd
import numpy as np

RAINFALL_CSV = "rainfall_hourly_multistation.csv"
RIVER_CSV = "river_level_hourly_multistation.csv"
FLOOD_EVIDENCE_CSV = "flood_evidence_bharatpur.csv"

WARNING_THRESHOLD_1H = 30
MODERATE_MIN, MODERATE_MAX = 5, 30
N_MODERATE_SAMPLE = 120
N_NORMAL_SAMPLE = 100

# ----------------------------------------------------------------------
# 1. Load rainfall -- mixed datetime formats need format='mixed'
# ----------------------------------------------------------------------
rf = pd.read_csv(RAINFALL_CSV, dtype={"station_id": str})
rf["datetime"] = pd.to_datetime(rf["datetime"], format="mixed", utc=True)
rf["event_time"] = rf["datetime"].dt.floor("h")  # align stations to same hour
rf = rf.rename(columns={"station_id": "rainfall_station_id"})

hourly = rf.groupby("event_time").agg(
    max_1h=("rainfall_1h_mm", "max"),
    max_score=("rainfall_score", "max"),
).reset_index()
print(f"Distinct hours with data: {len(hourly)}")

warning = hourly[hourly["max_1h"] >= WARNING_THRESHOLD_1H]
moderate = hourly[(hourly["max_1h"] >= MODERATE_MIN) & (hourly["max_1h"] < MODERATE_MAX)]
normal = hourly[hourly["max_1h"] < MODERATE_MIN]
print(f"Warning hours: {len(warning)}, Moderate: {len(moderate)}, Normal: {len(normal)}")

mod_sample = moderate.sample(n=min(N_MODERATE_SAMPLE, len(moderate)), random_state=42)
norm_sample = normal.sample(n=min(N_NORMAL_SAMPLE, len(normal)), random_state=42)

# ----------------------------------------------------------------------
# 2. Force-include flood incident dates -- searches peak RAINFALL hour
#    AND peak RIVER LEVEL hour separately, since for Narayani (large
#    upstream catchment from Pokhara/Annapurna region), the river's danger
#    crossing can lag local rainfall by many hours -- the actual flood
#    peak is often a RIVER LEVEL event, not a local rainfall event. Both
#    candidate hours get included so the dataset captures whichever
#    condition (rain or river) was actually most severe.
# ----------------------------------------------------------------------
ev = pd.read_csv(FLOOD_EVIDENCE_CSV, parse_dates=["event_date"])
ev_dates = pd.to_datetime(ev["event_date"]).dt.tz_localize("UTC").drop_duplicates()
hi = hourly.set_index("event_time").sort_index()

# Load river data now (needed here, not just later) to search for peak level
try:
    riv_early = pd.read_csv(RIVER_CSV)
    riv_early["datetime"] = pd.to_datetime(riv_early["datetime"], format="mixed", utc=True)
    riv_early["event_time"] = riv_early["datetime"].dt.floor("h")
    riv_hourly_max = riv_early.groupby("event_time")["water_level_m"].max()
except FileNotFoundError:
    riv_hourly_max = pd.Series(dtype=float)

peak_rows = []
for d in ev_dates:
    window_start, window_end = d - pd.Timedelta(days=2), d + pd.Timedelta(days=2)

    # Candidate 1: peak rainfall hour in window
    rain_window = hi[(hi.index >= window_start) & (hi.index <= window_end)]
    if len(rain_window):
        peak_rain_hour = rain_window["max_1h"].idxmax()
        peak_rows.append(hi.loc[peak_rain_hour])
        print(f"Evidence date {d.date()}: peak RAINFALL hour at {peak_rain_hour} "
              f"(max_1h={rain_window.loc[peak_rain_hour,'max_1h']:.1f}mm)")

    # Candidate 2: peak river level hour in window (may be a different hour
    # than peak rainfall -- this is the fix for upstream-lag events)
    if len(riv_hourly_max):
        riv_window = riv_hourly_max[(riv_hourly_max.index >= window_start) & (riv_hourly_max.index <= window_end)]
        if len(riv_window):
            peak_riv_hour = riv_window.idxmax()
            if peak_riv_hour in hi.index:
                peak_rows.append(hi.loc[peak_riv_hour])
                print(f"Evidence date {d.date()}: peak RIVER LEVEL hour at {peak_riv_hour} "
                      f"(water_level={riv_window.loc[peak_riv_hour]:.2f}m)")

flood_rows = pd.DataFrame(peak_rows).reset_index().rename(columns={"index": "event_time"}) if peak_rows else pd.DataFrame(columns=["event_time"])
flood_rows = flood_rows.drop_duplicates(subset="event_time")

# ----------------------------------------------------------------------
# 2b. NEW TIER: river-elevated-but-local-rain-low hours. This captures
#     upstream flooding scenarios (rain falling in Gandaki
#     province/upper catchment raises Narayani at Devghat with a lag,
#     while Bharatpur itself sees little local rain) -- a real,
#     documented, recurring pattern that pure rainfall-based tiers
#     above never systematically search for. Without this, the model
#     never sees training examples of "river already dangerous, no
#     local rain" and its behavior on that combination is unverified.
# ----------------------------------------------------------------------
try:
    riv_check = pd.read_csv(RIVER_CSV)
    riv_check["datetime"] = pd.to_datetime(riv_check["datetime"], format="mixed", utc=True)
    riv_check["event_time"] = riv_check["datetime"].dt.floor("h")
    riv_devghat_max = riv_check[riv_check["station_id"] == "RV01"].groupby("event_time")["water_level_m"].max()

    combined_check = hourly.set_index("event_time").join(riv_devghat_max, how="inner")
    river_flood_low_rain = combined_check[
        (combined_check["max_1h"] < 2) & (combined_check["water_level_m"] >= 7.3)
    ]
    n_found = len(river_flood_low_rain)
    print(f"\nFound {n_found} hours with river >= warning level AND local rainfall < 2mm "
          f"(upstream-flood scenario) -- sampling up to 30 for training inclusion")
    river_flood_sample = river_flood_low_rain.sample(n=min(30, n_found), random_state=42).reset_index()
except (FileNotFoundError, KeyError):
    river_flood_sample = pd.DataFrame(columns=["event_time"])
    print("NOTE: could not check for upstream-flood scenarios (river data unavailable at this stage)")

# ----------------------------------------------------------------------
# 3. Combine and save
# ----------------------------------------------------------------------
selected = pd.concat([warning, mod_sample, norm_sample, flood_rows, river_flood_sample], ignore_index=True)
selected = selected.drop_duplicates(subset="event_time").sort_values("event_time").reset_index(drop=True)
selected["event_id"] = [f"EVT{i+1:04d}" for i in range(len(selected))]
print(f"\nFinal selected events: {len(selected)}")
selected[["event_id", "event_time"]].to_csv("selected_events.csv", index=False)

# ----------------------------------------------------------------------
# 4. Per-station rainfall at selected events (mean if duplicate readings
#    within an hour after flooring)
# ----------------------------------------------------------------------
rain_cols = ["rainfall_1h_mm", "rainfall_3h_mm", "rainfall_6h_mm",
             "rainfall_12h_mm", "rainfall_24h_mm", "rainfall_score"]
rf_hourly = rf.groupby(["rainfall_station_id", "event_time"])[rain_cols].mean().reset_index()
rf_selected = rf_hourly.merge(selected[["event_id", "event_time"]], on="event_time", how="inner")
rf_selected.to_csv("rainfall_by_station_selected.csv", index=False)
print(f"Saved rainfall_by_station_selected.csv: {len(rf_selected)} rows")

# ----------------------------------------------------------------------
# 5. River level at selected events (same datetime handling)
# ----------------------------------------------------------------------
try:
    riv = pd.read_csv(RIVER_CSV)
    riv["datetime"] = pd.to_datetime(riv["datetime"], format="mixed", utc=True)
    riv["event_time"] = riv["datetime"].dt.floor("h")
    riv = riv.rename(columns={"station_id": "river_station_id"})
    riv_hourly = riv.groupby(["river_station_id", "event_time"]).agg(
        water_level_m=("water_level_m", "mean"),
        river_level_score=("river_level_score", "mean"),
    ).reset_index()
    riv_selected = riv_hourly.merge(selected[["event_id", "event_time"]], on="event_time", how="inner")
    riv_selected.to_csv("river_level_by_station_selected.csv", index=False)
    print(f"Saved river_level_by_station_selected.csv: {len(riv_selected)} rows")
except FileNotFoundError:
    print(f"NOTE: {RIVER_CSV} not found -- skipping river level for now.")