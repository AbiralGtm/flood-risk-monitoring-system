"""
09_build_historical_grid_training_rows.py

PURPOSE
-------
Takes your historical_reconstructed_1993_2020_overridden.csv (station/
gauge-level, daily resolution, 1993-2020) and blends it across all 2,434
grids using the EXACT SAME IDW logic 03_build_ml_training_dataset.py
already uses for your real hourly BIPAD data -- so historical rows are
computed consistently with everything else in your training set, not
via a separate ad-hoc method.

FIX APPLIED: your historical_reconstructed file's event_id was built
PER-STATION ("HIST_<date>_<station_id>"), which breaks IDW blending --
03's logic needs ONE shared event_id per calendar date so all 7
stations' rainfall for that date get pooled and IDW-weighted together,
the same way a single real BIPAD hour pools readings across stations.
This script rebuilds event_id from the date alone before blending.

RIVER DATA LIMITATION (documented, not a bug): the historical
reconstruction only has a calibrated stage curve for Devghat/Narayani
(RV01) -- there is no equivalent for Rajaiya/Rapti (RV02), consistent
with your project's own documented limitation (Rajaiya is an upstream
signal, not a local proxy, and 62km from Bharatpur). Grids assigned to
RV02 will get dynamic_hazard_score computed from rainfall alone for
these historical rows, matching 03's own fallback behavior when river
data is unavailable.

INPUT
------
static_grid_dataset_with_stations.csv   (your real 2,434-grid file,
                                          output of 00_assign_stations_to_grid.py)
grid_rainfall_idw_weights.csv           (output of 00)
historical_reconstructed_1993_2020_overridden.csv  (output of 06 + 08)

OUTPUT
------
historical_grid_training_rows.csv
  Same core schema as ml_training_dataset.csv (minus risk_class/
  flood_label, which get computed globally in the next step), PLUS:
    data_source, temporal_resolution, label_source, override_evidence_id
  carried through for transparency and later feature-importance checks.
"""

import sys
import numpy as np
import pandas as pd

STATIC_CSV = "static_grid_dataset_with_stations.csv"
WEIGHTS_CSV = "grid_rainfall_idw_weights.csv"
HISTORICAL_CSV = "historical_reconstructed_1993_2020_sampled.csv"
OUT_CSV = "historical_grid_training_rows.csv"

RAIN_COLS = ["rainfall_1h_mm", "rainfall_3h_mm", "rainfall_6h_mm",
             "rainfall_12h_mm", "rainfall_24h_mm", "rainfall_score"]


def main():
    try:
        static = pd.read_csv(STATIC_CSV)
    except FileNotFoundError:
        print(f"MISSING INPUT: {STATIC_CSV}")
        sys.exit(1)
    try:
        weights = pd.read_csv(WEIGHTS_CSV, dtype={"rainfall_station_id": str})
    except FileNotFoundError:
        print(f"MISSING INPUT: {WEIGHTS_CSV}")
        sys.exit(1)
    try:
        hist = pd.read_csv(HISTORICAL_CSV)
        # Use format='mixed' + utc=True, matching your own established
        # convention in 02_rainfall_event_selection.py -- BIPAD/reconstructed
        # timestamps can have subtly inconsistent formats (with/without
        # microseconds, different tz representations), and pandas' default
        # parse_dates auto-inference can fail outright or silently mis-parse
        # in ways that break downstream date-based grouping.
        hist["event_time"] = pd.to_datetime(hist["event_time"], format="mixed", utc=True)
    except FileNotFoundError:
        print(f"MISSING INPUT: {HISTORICAL_CSV}")
        sys.exit(1)

    print(f"Loaded {len(static)} grids, {len(weights)} IDW weight rows, "
          f"{len(hist)} historical station-rows.")

    # ------------------------------------------------------------------
    # FIX: rebuild event_id from the calendar date alone, so all stations
    # reporting for the same date share one event_id (required for IDW
    # blending -- mirrors how real BIPAD hourly events work in 03).
    # ------------------------------------------------------------------
    hist = hist.copy()
    hist["date_only"] = hist["event_time"].dt.date
    hist["event_id"] = "HIST_" + hist["date_only"].astype(str)
    hist["rainfall_station_id"] = hist["rainfall_station_id"].astype(str)

    n_unique_dates = hist["date_only"].nunique()
    n_unique_old_event_ids = hist.groupby("date_only")["rainfall_station_id"].nunique()
    print(f"Rebuilt event_id from date alone: {n_unique_dates} unique calendar dates, "
          f"avg {n_unique_old_event_ids.mean():.1f} stations pooled per date "
          f"(should be close to 7).")

    # ------------------------------------------------------------------
    # Vectorized IDW blend -- IDENTICAL logic to 03_build_ml_training_dataset.py
    # ------------------------------------------------------------------
    stations = sorted(weights["rainfall_station_id"].unique())
    s_idx = {s: i for i, s in enumerate(stations)}
    grids = static["grid_id"].values
    g_idx = {g: i for i, g in enumerate(grids)}

    unknown = set(hist["rainfall_station_id"].unique()) - set(stations)
    if unknown:
        before_n = len(hist)
        hist = hist[hist["rainfall_station_id"].isin(stations)].copy()
        print(f"WARNING: dropped {before_n - len(hist)} historical rows for "
              f"station(s) {sorted(unknown)} not present in rainfall_stations.csv.")

    W = np.zeros((len(grids), len(stations)))
    for _, r in weights.iterrows():
        if r["grid_id"] in g_idx:
            W[g_idx[r["grid_id"]], s_idx[r["rainfall_station_id"]]] = r["weight"]

    events = sorted(hist["event_id"].unique())
    e_idx = {e: i for i, e in enumerate(events)}

    avail = np.zeros((len(stations), len(events)), dtype=bool)
    V = {c: np.zeros((len(stations), len(events))) for c in RAIN_COLS}
    for _, r in hist.iterrows():
        si, ei = s_idx[r["rainfall_station_id"]], e_idx[r["event_id"]]
        avail[si, ei] = True
        for c in RAIN_COLS:
            val = r[c] if c in hist.columns else np.nan
            V[c][si, ei] = val if pd.notna(val) else 0.0

    print("Computing IDW-blended historical rainfall (vectorized)...")
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

    # FIX: build ev_meta by explicit groupby-first on event_id ALONE, not a
    # multi-column drop_duplicates(). The previous version relied on
    # event_time/data_source/temporal_resolution being byte-identical
    # across all 7 stations for the same date to collapse correctly --
    # if even one of those columns had ANY per-row variation (e.g. a
    # subtle timestamp precision difference), drop_duplicates() would
    # silently leave MULTIPLE rows per event_id, and the merge below
    # would fan out into a cartesian product -- multiplying every grid's
    # row count by however many near-duplicate event_id rows existed.
    # This is almost certainly what caused a 16GB output file. Explicit
    # groupby-first guarantees exactly one row per event_id, and the
    # assertion after makes this loud and immediate if it's ever violated
    # again, instead of silently producing a multi-gigabyte file.
    ev_meta = hist.groupby("event_id").first()[
        ["event_time", "data_source", "temporal_resolution"]
    ].reset_index()

    if "label_source" in hist.columns:
        label_meta = hist.groupby("event_id").agg(
            label_source=("label_source", lambda x: "historical_evidence_override"
                          if (x == "historical_evidence_override").any() else "reconstructed"),
            override_evidence_id=("override_evidence_id", lambda x: "; ".join(
                sorted(set(str(v) for v in x if str(v).strip())))),
            # Preserve the override's floor value (max across stations for that
            # date -- 08's override sets all stations to the same floor, so max
            # correctly recovers it) so it can be RE-APPLIED after recomputing
            # dynamic_hazard_score fresh below, rather than lost entirely.
            override_floor_value=("dynamic_hazard_score", "max"),
        ).reset_index()
        ev_meta = ev_meta.merge(label_meta, on="event_id", how="left")

    assert ev_meta["event_id"].is_unique, (
        f"ev_meta has duplicate event_id rows ({len(ev_meta) - ev_meta['event_id'].nunique()} "
        f"extra) -- merging this would explode row count. This should be impossible after "
        f"groupby('event_id').first(); if you see this, something upstream is producing "
        f"inconsistent event_id values (check for whitespace/type differences in the "
        f"historical file's event_id or date columns)."
    )
    print(f"ev_meta: {len(ev_meta):,} rows, verified exactly one per event_id.")

    n_before_merge = len(grid_rain)
    grid_rain = grid_rain.merge(ev_meta, on="event_id")
    if len(grid_rain) != n_before_merge:
        print(f"WARNING: row count changed during ev_meta merge "
              f"({n_before_merge:,} -> {len(grid_rain):,}) -- this should not happen "
              f"with a clean 1-to-1 merge, investigate before proceeding.")
    print(f"Blended historical rainfall rows: {len(grid_rain):,}")

    # ------------------------------------------------------------------
    # Join to static grid attributes
    # ------------------------------------------------------------------
    ml = static.merge(grid_rain, on="grid_id", how="inner")
    print(f"After static join: {len(ml):,} rows")

    # ------------------------------------------------------------------
    # River level -- ONLY for RV01 (Devghat/Narayani) grids, per the
    # documented limitation that no Rajaiya stage curve exists.
    #
    # FIX (confirmed via real-data investigation): dynamic_hazard_score
    # must be RECOMPUTED here using each grid's own IDW-blended
    # rainfall_score, NOT copied from the historical station-level file.
    # The historical file's dynamic_hazard_score was computed pre-blend
    # (dominated by whichever single station had the most extreme
    # reading that day) -- copying it directly meant every grid for a
    # given date got the SAME stale value regardless of that grid's own
    # actual blended rainfall, which silently classified genuinely calm
    # grids (low blended rainfall) as High/Danger because a DIFFERENT,
    # rainier station's old reading leaked through unchanged. Confirmed
    # empirically: 1998-06-29 showed dynamic_hazard_score=0.770911
    # identically across 217 different grids with 217 different blended
    # rainfall_score values -- a value that can't vary per grid is not
    # actually being computed per grid.
    # ------------------------------------------------------------------
    river_cols_needed = ["event_id", "water_level_m", "river_level_score"]
    missing_river_cols = [c for c in river_cols_needed if c not in hist.columns]
    if missing_river_cols:
        print(f"WARNING: historical file missing {missing_river_cols} -- "
              f"river-floor logic will be skipped, using rainfall-only hazard.")
        ml["dynamic_hazard_score"] = ml["rainfall_score"]
        ml["water_level_m"] = np.nan
        ml["river_level_score"] = np.nan
    else:
        riv_slim = hist[river_cols_needed].drop_duplicates(subset="event_id")
        assert riv_slim["event_id"].is_unique, "riv_slim has duplicate event_id -- would explode the merge below."

        is_rv01 = ml["river_station_id"] == "RV01"
        print(f"Grids assigned to RV01 (Narayani/Devghat, HAS historical river data): "
              f"{is_rv01.sum():,} rows")
        print(f"Grids assigned to RV02 (Rapti/Rajaiya, NO historical river data): "
              f"{(~is_rv01).sum():,} rows -- these get rainfall-only hazard for "
              f"historical events, per the documented Rajaiya limitation.")

        ml = ml.merge(riv_slim, on="event_id", how="left", suffixes=("", "_hist"))
        # For RV02 grids, no river data applies at all.
        ml.loc[~is_rv01, ["water_level_m", "river_level_score"]] = np.nan

        # Recompute dynamic_hazard_score fresh, per grid, using THIS grid's
        # own blended rainfall_score -- identical river-floor formula to 03.
        weighted_avg = 0.70 * ml["rainfall_score"] + 0.30 * ml["river_level_score"]
        river_floor = ml["river_level_score"].fillna(0)
        ml["dynamic_hazard_score"] = np.where(
            ml["river_level_score"].notna(),
            np.maximum(weighted_avg, river_floor),
            ml["rainfall_score"],
        )

        # Re-apply the evidence-override floor on TOP of the freshly
        # recomputed value, only for override-flagged rows -- this
        # preserves 08's intentional force-override (known flood dates
        # should classify as severe regardless of that day's blended
        # rainfall/river reading) while still fixing the stale-value bug
        # for every non-override row.
        if "label_source" in ml.columns and "override_floor_value" in ml.columns:
            is_override = ml["label_source"] == "historical_evidence_override"
            n_before = ml.loc[is_override, "dynamic_hazard_score"].copy()
            ml.loc[is_override, "dynamic_hazard_score"] = np.maximum(
                ml.loc[is_override, "dynamic_hazard_score"],
                ml.loc[is_override, "override_floor_value"],
            )
            print(f"Re-applied evidence-override floor to {is_override.sum():,} rows "
                  f"(preserving 08's intentional force-override after the blend fix).")

    # ------------------------------------------------------------------
    # FINAL RISK SCORE -- identical multiplicative formula to 03
    # ------------------------------------------------------------------
    ml["final_risk_score"] = ml["static_susceptibility_score"] * (0.2 + 0.8 * ml["dynamic_hazard_score"])

    if "nearest_major_river" in ml.columns:
        river_map = {r: i for i, r in enumerate(sorted(ml["nearest_major_river"].dropna().unique()))}
        ml["nearest_major_river_encoded"] = ml["nearest_major_river"].map(river_map)

    for c in ["within_2km_narayani", "within_2km_rapti"]:
        if c in ml.columns and ml[c].dtype == object:
            ml[c] = ml[c].astype(str).str.upper().eq("TRUE").astype(int)

    ml.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(ml):,} historical grid-level rows to {OUT_CSV}")
    print("\n--- SELF-VERIFICATION ---")
    checks_passed = True
    if ml["final_risk_score"].between(0, 1).all():
        print("PASS: final_risk_score within [0,1]")
    else:
        print("FAIL: final_risk_score out of range")
        checks_passed = False
    if ml["grid_id"].nunique() == static["grid_id"].nunique():
        print(f"PASS: all {static['grid_id'].nunique()} grids represented")
    else:
        print(f"WARNING: only {ml['grid_id'].nunique()} / {static['grid_id'].nunique()} "
              f"grids appear in the historical output -- check IDW weights coverage")
    print(f"\nOverall: {'PASS' if checks_passed else 'FAIL -- review warnings above'}")
    print("\nNEXT STEP: run 10_merge_and_finalize_training_data.py to concatenate this")
    print("with your existing ml_training_dataset.csv and recompute risk_class")
    print("thresholds over the COMBINED dataset.")


if __name__ == "__main__":
    main()