"""
06_chirps_glofas_historical_backfill.py

PURPOSE
-------
Reconstructs rainfall (CHIRPS) and river severity (GloFAS-ERA5 discharge,
converted to an approximate stage/severity score via anchor-point
calibration) for 1993-2020, so your known historical flood years
(1993, 2000, 2008, 2010, 2011) and evidence records EV024/EV025 (2020)
can become full training rows instead of history_score-only entries.

UPDATE: column-name matching corrected against your REAL schema, now
confirmed from 02_rainfall_event_selection.py:
  rainfall_hourly_multistation.csv:
    columns: station_id, datetime (MIXED format -- some rows have
    microseconds, some don't, mixed tz representations -- must use
    pd.to_datetime(..., format='mixed', utc=True), same as 02 does),
    rainfall_1h_mm, rainfall_3h_mm, rainfall_6h_mm, rainfall_12h_mm,
    rainfall_24h_mm, rainfall_score
  river_level_hourly_multistation.csv:
    columns: station_id, datetime (same mixed-format issue),
    water_level_m, river_level_score

RUN THIS LOCALLY / ON COLAB. The CHIRPS/CDS/DHM data hosts are not
reachable from the Claude sandbox network.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
- It does NOT give you a real measured river stage. There is no published
  Devghat rating curve (confirmed via research -- DHM holds this
  internally). This script builds an approximate stage-severity score
  from GloFAS discharge using anchor points, carries an explicit
  uncertainty flag, and tags every reconstructed row so you can exclude
  or downweight it later if it hurts model quality.
- It does NOT let you mix CHIRPS v2 and v3 in one series -- pick one and
  stay consistent across the whole 1993-2026 span (v3 is systematically
  wetter by construction).
- It ONLY builds an approximate stage curve for Devghat/Narayani (RV01).
  There is no equivalent anchor-point data for Rajaiya/Rapti (RV02) --
  if you need historical Rapti severity too, that gauge's reconstructed
  river_level_score will fall back to the 0.3 default used elsewhere in
  your pipeline for unknown conditions, which is NOT a real estimate.
  Flag any Rapti-side reconstructed rows as lower-confidence in your report.

REQUIRED INPUTS
----------------
1. CHIRPS daily data (v3 recommended), 1993-01-01 to 2020-12-31, clipped
   to your 7 rainfall station locations (rainfall_stations.csv).
   Get from: https://data.chc.ucsb.edu/products/CHIRPS/v3.0/
   or via Google Earth Engine (UCSB-CHG/CHIRPS/DAILY collection).
   Expected local file: chirps_daily_1993_2020.csv
     columns: date, station_id, rainfall_24h_mm

2. GloFAS-ERA5 discharge reanalysis, 1993-01-01 to 2020-12-31, at grid
   cells nearest Devghat and Rajaiya.
   Get from: Copernicus Climate Data Store (cds.climate.copernicus.eu),
   dataset "cems-glofas-historical" -- requires a free CDS API key.
   Expected local file: glofas_discharge_1993_2020.csv
     columns: date, gauge_id, discharge_m3s
   (gauge_id should be "RV01" for Devghat/Narayani, "RV02" for Rajaiya/Rapti,
   matching your existing river_stations.csv station_id values)

3. Your existing hourly data (for bias-correction reference overlap):
   rainfall_hourly_multistation.csv
   river_level_hourly_multistation.csv

OUTPUT
------
historical_reconstructed_1993_2020.csv
  Columns match your rainfall_by_station_selected.csv /
  river_level_by_station_selected.csv schema as closely as possible, so
  it can be concatenated onto those files before re-running
  03_build_ml_training_dataset.py. See INTEGRATION NOTE at the bottom of
  this script's output for the exact steps -- this script does NOT
  automatically merge into your existing training pipeline, since that
  merge should only happen after you've manually verified the
  reconstructed rows (per the research report's recommendation).

ANCHOR POINTS USED FOR DISCHARGE -> STAGE CALIBRATION (Devghat/Narayani only)
--------------------------------------------------------------------------
  ~8,900  m3/s  ->  7.3-7.5 m  (warning threshold, two sources disagree
                                 slightly -- averaged here, +/-0.5m band)
  ~10,700 m3/s  ->  8.3-9.0 m  (danger threshold, same caveat, +/-0.7m band)
  ~11,000-12,100 m3/s -> ~10-10.75 m (2024 observed historic peak range,
                                       used only to anchor the curve's
                                       upper tail, +/-1.0m band)
These are NOT a measured rating curve. Treat this as your best available
proxy until/unless DHM provides the real station-450 rating table.
"""

import sys
import numpy as np
import pandas as pd


# --- Anchor points for the discharge -> stage power-law fit (Narayani @ Devghat) ---
ANCHOR_DISCHARGE_M3S = np.array([0, 8900, 10700, 11500])
ANCHOR_STAGE_M       = np.array([0, 7.4, 8.65, 10.3])   # midpoints of the ranges above
ANCHOR_UNCERTAINTY_M = np.array([0.3, 0.5, 0.7, 1.0])   # +/- band, grows with discharge

# Devghat operational thresholds (from your existing river_stations.csv)
DEVGHAT_WARNING_M = 7.3
DEVGHAT_DANGER_M = 9.0
DEVGHAT_GAUGE_ID = "RV01"   # matches RIVER_TO_GAUGE mapping in 00_assign_stations_to_grid.py
RAJAIYA_GAUGE_ID = "RV02"


def fit_stage_curve():
    """
    Fits a simple monotonic power-law-style curve through the anchor points
    using log-space linear interpolation (safer than a raw power-law fit
    with only 4 points -- avoids wild extrapolation beyond the anchors).
    Returns a function discharge_to_stage(q) -> (stage_m, uncertainty_m).
    ONLY valid for Devghat/Narayani -- see module docstring.
    """
    log_q = np.log1p(ANCHOR_DISCHARGE_M3S)

    def discharge_to_stage(q):
        q = max(q, 0)
        lq = np.log1p(q)
        stage = np.interp(lq, log_q, ANCHOR_STAGE_M)
        uncertainty = np.interp(lq, log_q, ANCHOR_UNCERTAINTY_M)
        if q > ANCHOR_DISCHARGE_M3S[-1]:
            uncertainty *= 1.5
        return stage, uncertainty

    return discharge_to_stage


def river_level_score_from_stage(stage_m, warning=DEVGHAT_WARNING_M, danger=DEVGHAT_DANGER_M):
    """
    Mirrors the tiered river_level_score logic implied by
    03_build_ml_training_dataset.py's river-floor fix, so reconstructed
    rows are scored consistently with your real hourly rows.

    FIX: the sub-warning band (70%-100% of warning) used to return a
    FLAT 0.3 for the entire range -- confirmed via real 1993-2020 data
    to cause multiple genuinely-different years (1993, 1994, 2008, 2011,
    2012-2019) to tie at an IDENTICAL dynamic_hazard_score (0.79000),
    masking real relative severity differences between them. Now
    continuous across that band (0.1 at 70% of warning, ramping smoothly
    to 0.3 at the warning threshold itself), matching the style already
    used for the warning-danger and above-danger tiers. Verified
    continuous at both boundaries (no new discontinuities introduced).
    """
    low_bound = warning * 0.7
    if stage_m < low_bound:
        return 0.1
    elif stage_m < warning:
        return 0.1 + 0.2 * (stage_m - low_bound) / (warning - low_bound)
    elif stage_m < danger:
        return 0.3 + 0.4 * (stage_m - warning) / (danger - warning)
    else:
        excess = min((stage_m - danger) / (danger * 0.3), 1.0)
        return 0.7 + 0.3 * excess


def rainfall_score_from_24h(rainfall_24h_mm):
    """
    Reconstructed rows only have a daily (24h) CHIRPS value -- no 1h/3h/6h/12h
    breakdown. This uses ONLY the 24h tier of your documented formula:
        rainfall_score = min(1.0, max(1h/60, 3h/80, 6h/100, 12h/120, 24h/140))
    Since sub-daily tiers are unavailable, this will systematically
    UNDER-represent short, intense bursts that your real hourly data would
    catch. Document this limitation in your report -- it is a genuine and
    unavoidable resolution gap, not a bug.
    """
    return min(1.0, rainfall_24h_mm / 140.0)


def quantile_bias_correct(historical_series, observed_reference_series):
    """
    Simple quantile mapping: maps each value in historical_series to the
    quantile it would represent in observed_reference_series's distribution.
    """
    historical_series = pd.Series(historical_series).dropna()
    observed_reference_series = pd.Series(observed_reference_series).dropna()

    if len(observed_reference_series) == 0 or len(historical_series) == 0:
        print("WARNING: empty series passed to bias correction, returning uncorrected values")
        return historical_series

    ref_sorted = np.sort(observed_reference_series.values)
    hist_ranks = historical_series.rank(pct=True).values
    corrected = np.interp(hist_ranks, np.linspace(0, 1, len(ref_sorted)), ref_sorted)
    return pd.Series(corrected, index=historical_series.index)


def load_observed_daily_rainfall(path):
    """
    Loads rainfall_hourly_multistation.csv (real schema: station_id,
    datetime [mixed format], rainfall_1h_mm...rainfall_24h_mm, rainfall_score)
    and aggregates to one row per station-day for bias-correction reference.
    """
    df = pd.read_csv(path, dtype={"station_id": str})
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", utc=True)
    df["date"] = df["datetime"].dt.date
    # rainfall_24h_mm at any given hour is already a rolling 24h total in
    # your schema (per 02's usage) -- take the daily MAX per station-day
    # as the day's characteristic 24h accumulation, not a sum (summing
    # rolling totals would double count).
    daily = df.groupby(["station_id", "date"])["rainfall_24h_mm"].max().reset_index()
    return daily


def load_observed_river_level(path, gauge_id):
    """
    Loads river_level_hourly_multistation.csv (real schema: station_id,
    datetime [mixed format], water_level_m, river_level_score) filtered
    to one gauge, aggregated to daily max.
    """
    df = pd.read_csv(path, dtype={"station_id": str})
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", utc=True)
    df["date"] = df["datetime"].dt.date
    df = df[df["station_id"] == gauge_id]
    return df.groupby("date")["water_level_m"].max()


def main():
    CHIRPS_PATH = "chirps_daily_1993_2020.csv"
    GLOFAS_PATH = "glofas_discharge_1993_2020.csv"
    OBSERVED_RAINFALL_PATH = "rainfall_hourly_multistation.csv"
    OBSERVED_RIVER_PATH = "river_level_hourly_multistation.csv"
    OUT_PATH = "historical_reconstructed_1993_2020.csv"

    for path in [CHIRPS_PATH, GLOFAS_PATH, OBSERVED_RAINFALL_PATH, OBSERVED_RIVER_PATH]:
        try:
            open(path).close()
        except FileNotFoundError:
            print(f"MISSING INPUT: {path}")
            print("See the docstring at the top of this script for where to get it.")
            sys.exit(1)

    print("Loading CHIRPS daily rainfall ...")
    chirps = pd.read_csv(CHIRPS_PATH, parse_dates=["date"], dtype={"station_id": str})
    chirps["date"] = chirps["date"].dt.date

    print("Loading GloFAS-ERA5 discharge ...")
    glofas = pd.read_csv(GLOFAS_PATH, parse_dates=["date"], dtype={"gauge_id": str})
    glofas["date"] = glofas["date"].dt.date
    glofas_devghat = glofas[glofas["gauge_id"] == DEVGHAT_GAUGE_ID].copy()
    n_rajaiya = (glofas["gauge_id"] == RAJAIYA_GAUGE_ID).sum()
    if n_rajaiya:
        print(f"NOTE: {n_rajaiya} Rajaiya/Rapti (RV02) discharge rows found in GloFAS input, "
              f"but NO stage-calibration anchors exist for that gauge. These will fall back "
              f"to the 0.3 default river_level_score -- flag as low-confidence if used.")

    print("Loading observed hourly rainfall for bias-correction reference ...")
    obs_daily_rain = load_observed_daily_rainfall(OBSERVED_RAINFALL_PATH)

    print("Loading observed hourly river level (Devghat) for context ...")
    obs_daily_river = load_observed_river_level(OBSERVED_RIVER_PATH, DEVGHAT_GAUGE_ID)
    if len(obs_daily_river):
        print(f"  Observed Devghat water_level_m range: "
              f"{obs_daily_river.min():.2f}m - {obs_daily_river.max():.2f}m "
              f"({len(obs_daily_river)} days)")

    # --- Bias-correct CHIRPS against observed daily-aggregated rainfall ---
    print("\nBias-correcting CHIRPS (quantile mapping per station) ...")
    corrected_chirps = []
    for station_id, group in chirps.groupby("station_id"):
        ref = obs_daily_rain[obs_daily_rain["station_id"] == station_id]["rainfall_24h_mm"]
        if len(ref) < 30:
            print(f"  Station {station_id}: insufficient overlap ({len(ref)} days) -- "
                  f"using uncorrected CHIRPS, flag this row set as lower confidence")
            group = group.copy()
            group["rainfall_24h_mm_corrected"] = group["rainfall_24h_mm"]
            group["bias_corrected"] = False
        else:
            group = group.copy()
            group["rainfall_24h_mm_corrected"] = quantile_bias_correct(
                group["rainfall_24h_mm"], ref
            ).values
            group["bias_corrected"] = True
        corrected_chirps.append(group)
    chirps_corrected = pd.concat(corrected_chirps, ignore_index=True)

    # --- Discharge -> stage (Devghat/Narayani only) ---
    print("Converting Devghat discharge to stage estimate via anchor-point curve ...")
    stage_fn = fit_stage_curve()
    glofas_devghat[["stage_estimate_m", "stage_uncertainty_m"]] = glofas_devghat["discharge_m3s"].apply(
        lambda q: pd.Series(stage_fn(q))
    )

    # --- Compute scores exactly following your documented formulas ---
    print("Computing rainfall_score, river_level_score, dynamic_hazard_score ...")
    chirps_corrected["rainfall_score"] = chirps_corrected["rainfall_24h_mm_corrected"].apply(
        rainfall_score_from_24h
    )
    glofas_devghat["river_level_score"] = glofas_devghat["stage_estimate_m"].apply(
        river_level_score_from_stage
    )

    merged = chirps_corrected.merge(
        glofas_devghat[["date", "river_level_score", "stage_estimate_m", "stage_uncertainty_m"]],
        on="date", how="left"
    )
    merged["river_level_score"] = merged["river_level_score"].fillna(0.3)  # documented fallback

    # river-floor fix, exactly as in 03_build_ml_training_dataset.py
    merged["dynamic_hazard_score"] = np.maximum(
        0.70 * merged["rainfall_score"] + 0.30 * merged["river_level_score"],
        merged["river_level_score"]
    )

    # ------------------------------------------------------------------
    # Reshape to match rainfall_by_station_selected.csv's schema as closely
    # as possible, so it can be concatenated before re-running 03.
    # ------------------------------------------------------------------
    merged["event_time"] = pd.to_datetime(merged["date"]).dt.tz_localize("UTC")
    merged["event_id"] = "HIST_" + merged["date"].astype(str) + "_" + merged["station_id"]
    merged["rainfall_station_id"] = merged["station_id"]
    merged["rainfall_1h_mm"] = np.nan   # not available at daily resolution -- explicit NaN, not 0
    merged["rainfall_3h_mm"] = np.nan
    merged["rainfall_6h_mm"] = np.nan
    merged["rainfall_12h_mm"] = np.nan
    merged["rainfall_24h_mm"] = merged["rainfall_24h_mm_corrected"]
    merged["water_level_m"] = merged["stage_estimate_m"]
    merged["river_station_id"] = DEVGHAT_GAUGE_ID

    merged["data_source"] = "reanalysis_daily"
    merged["temporal_resolution"] = "daily"

    out_cols = [
        "event_id", "event_time", "rainfall_station_id",
        "rainfall_1h_mm", "rainfall_3h_mm", "rainfall_6h_mm",
        "rainfall_12h_mm", "rainfall_24h_mm", "rainfall_score",
        "river_station_id", "water_level_m", "river_level_score",
        "dynamic_hazard_score", "stage_uncertainty_m",
        "data_source", "temporal_resolution", "bias_corrected",
    ]
    merged[out_cols].to_csv(OUT_PATH, index=False)

    # --- Self-verification ---
    print("\n--- SELF-VERIFICATION ---")
    checks_passed = True

    if merged["dynamic_hazard_score"].between(0, 1).all():
        print("PASS: dynamic_hazard_score within [0,1]")
    else:
        print("FAIL: dynamic_hazard_score out of range")
        checks_passed = False

    uncorrected_frac = (~chirps_corrected["bias_corrected"]).mean()
    if uncorrected_frac > 0.3:
        print(f"WARNING: {100*uncorrected_frac:.0f}% of stations had insufficient overlap "
              f"for bias correction -- results for those stations are lower confidence")
    else:
        print(f"PASS: {100*(1-uncorrected_frac):.0f}% of stations bias-corrected successfully")

    n_extrapolated = (glofas_devghat["discharge_m3s"] > ANCHOR_DISCHARGE_M3S[-1]).sum()
    print(f"NOTE: {n_extrapolated} days had Devghat discharge above the highest anchor point "
          f"({ANCHOR_DISCHARGE_M3S[-1]} m3/s) -- these stage estimates carry 1.5x uncertainty")

    print(f"\nOverall: {'PASS' if checks_passed else 'FAIL -- review warnings above'}")
    print(f"Saved: {OUT_PATH}  ({len(merged):,} rows)")

    print("\n" + "="*70)
    print("INTEGRATION NOTE -- read before merging into your training pipeline")
    print("="*70)
    print("This file's rows are DAILY resolution, station-keyed like")
    print("rainfall_by_station_selected.csv, but are NOT yet grid-level like")
    print("ml_training_dataset.csv. To actually use them:")
    print("  1. Manually verify a sample of reconstructed rows against known")
    print("     flood dates (1993, 2000, 2008, 2010, 2011) before trusting them.")
    print("  2. Concatenate onto rainfall_by_station_selected.csv and")
    print("     river_level_by_station_selected.csv (matching event_id/event_time")
    print("     schema), OR run these historical event_ids through the same")
    print("     IDW-blend + static-join logic in 03_build_ml_training_dataset.py.")
    print("  3. Keep the data_source and temporal_resolution columns through to")
    print("     the final ml_training_dataset.csv, then check XGBoost feature")
    print("     importance for whether it leans on data_source -- if so, split")
    print("     into two model tiers instead of merging into one, per the")
    print("     earlier research report's recommendation.")


if __name__ == "__main__":
    main()