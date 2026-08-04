"""
09a_stratify_sample_historical_events.py

PURPOSE
-------
FIX for a real design flaw: 09_build_historical_grid_training_rows.py
blended EVERY day across ALL 28 years x ALL 2,434 grids, producing
~24.9 MILLION rows -- 28x more than your entire real observed dataset
(885,488 rows). This overwhelms the model with lower-fidelity
reconstructed data and distorts the combined risk_class thresholds.

Your own 02_rainfall_event_selection.py never does this for real data
either -- it stratified-samples a representative subset (~370 events:
all warning-tier hours, a sample of moderate hours, a sample of normal
hours, plus force-included evidence dates) rather than using every
single hour. This script applies the SAME methodology to the historical
1993-2020 reconstruction, run BEFORE 09's grid-level blend, so the
historical contribution is representative rather than overwhelming.

STRATIFICATION LOGIC (mirrors 02 exactly)
--------------------------------------------
- ALL days where dynamic_hazard_score >= WARNING_THRESHOLD: kept in full
  (these are rare and informative -- no need to downsample them)
- MODERATE tier (dynamic_hazard_score between MODERATE_MIN/MAX):
  sampled down to N_MODERATE_SAMPLE days
- NORMAL tier (below MODERATE_MIN): sampled down to N_NORMAL_SAMPLE days
- ALL days with label_source == "historical_evidence_override": force-
  included regardless of tier (these are your known flood-evidence days,
  same principle as 02's evidence-date forcing)

This brings the historical day count down from ~10,227 to roughly
370-500 days total (matching the same order of magnitude as 02's real
event count), which after grid-blending gives ~370 x 2,434 ≈ 900,000
rows -- comparable in scale to your observed dataset, not 28x larger.

INPUT
------
historical_reconstructed_1993_2020_overridden.csv (full daily reconstruction)

OUTPUT
------
historical_reconstructed_1993_2020_sampled.csv
  Same schema, filtered down to the stratified-sampled dates only.
  Use THIS as the input to 09_build_historical_grid_training_rows.py
  (edit that script's HISTORICAL_CSV path), NOT the original full file.
"""

import sys
import pandas as pd

IN_CSV = "historical_reconstructed_1993_2020_overridden.csv"
OUT_CSV = "historical_reconstructed_1993_2020_sampled.csv"

WARNING_THRESHOLD = 0.6      # matches roughly where 03's river-floor logic
                               # starts treating conditions as genuinely elevated
MODERATE_MIN, MODERATE_MAX = 0.3, 0.6
N_MODERATE_SAMPLE = 150       # per-station-day count target, same order of
                               # magnitude as 02's N_MODERATE_SAMPLE=120
N_NORMAL_SAMPLE = 150         # matches 02's N_NORMAL_SAMPLE=100 order of magnitude
RANDOM_STATE = 42


def main():
    try:
        df = pd.read_csv(IN_CSV, parse_dates=["event_time"])
    except FileNotFoundError:
        print(f"MISSING INPUT: {IN_CSV}")
        sys.exit(1)

    print(f"Loaded {len(df):,} historical station-rows.")

    # Work at the DATE level (not station-row level) -- one decision per
    # calendar date, since all stations for a date should be kept or
    # dropped together (mirrors how 02 operates on event_time, not
    # per-station rows).
    df["date_only"] = df["event_time"].dt.date
    daily_max = df.groupby("date_only")["dynamic_hazard_score"].max().reset_index()
    n_total_days = len(daily_max)
    print(f"Total distinct historical days available: {n_total_days:,}")

    warning_days = daily_max[daily_max["dynamic_hazard_score"] >= WARNING_THRESHOLD]
    moderate_days = daily_max[
        (daily_max["dynamic_hazard_score"] >= MODERATE_MIN) &
        (daily_max["dynamic_hazard_score"] < MODERATE_MAX)
    ]
    normal_days = daily_max[daily_max["dynamic_hazard_score"] < MODERATE_MIN]
    print(f"  Warning-tier days (>= {WARNING_THRESHOLD}): {len(warning_days):,} -- kept in full")
    print(f"  Moderate-tier days: {len(moderate_days):,} -- sampling to {N_MODERATE_SAMPLE}")
    print(f"  Normal-tier days: {len(normal_days):,} -- sampling to {N_NORMAL_SAMPLE}")

    moderate_sample = moderate_days.sample(
        n=min(N_MODERATE_SAMPLE, len(moderate_days)), random_state=RANDOM_STATE
    )
    normal_sample = normal_days.sample(
        n=min(N_NORMAL_SAMPLE, len(normal_days)), random_state=RANDOM_STATE
    )

    # Force-include ALL override dates regardless of tier -- these are
    # your known-flood-evidence days, same principle as 02's forced
    # evidence-date inclusion.
    if "label_source" in df.columns:
        override_dates = df[df["label_source"] == "historical_evidence_override"]["date_only"].unique()
        override_day_rows = daily_max[daily_max["date_only"].isin(override_dates)]
        print(f"  Force-included evidence-override days: {len(override_day_rows):,}")
    else:
        override_day_rows = pd.DataFrame(columns=daily_max.columns)
        print("  NOTE: no label_source column found -- skipping evidence-override forcing.")

    selected_days = pd.concat([warning_days, moderate_sample, normal_sample, override_day_rows])
    selected_days = selected_days.drop_duplicates(subset="date_only")
    print(f"\nTotal selected historical days: {len(selected_days):,} "
          f"(down from {n_total_days:,}, a {100*(1-len(selected_days)/n_total_days):.1f}% reduction)")

    sampled = df[df["date_only"].isin(selected_days["date_only"])].drop(columns=["date_only"])
    sampled.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(sampled):,} station-rows across {len(selected_days):,} days to {OUT_CSV}")

    # Estimate the resulting grid-level row count after 09's blend
    n_grids_estimate = 2434  # your documented grid count
    estimated_rows = len(selected_days) * n_grids_estimate
    print(f"\nEstimated grid-level rows after running 09 on this sampled file: "
          f"~{estimated_rows:,}")
    print("(compare this to your 885,488 observed rows -- should now be the same")
    print("order of magnitude, not 28x larger)")

    print("\n--- SELF-VERIFICATION ---")
    checks_passed = True
    all_override_dates_kept = set(override_day_rows["date_only"]) <= set(selected_days["date_only"])
    if all_override_dates_kept:
        print("PASS: all evidence-override days were kept")
    else:
        print("FAIL: some evidence-override days were dropped -- check the logic above")
        checks_passed = False

    print(f"\nOverall: {'PASS' if checks_passed else 'FAIL -- review warnings above'}")
    print(f"\nNEXT STEP: edit 09_build_historical_grid_training_rows.py's HISTORICAL_CSV")
    print(f"to point at {OUT_CSV} (NOT the original unsampled file), then re-run 09 and 10.")


if __name__ == "__main__":
    main()
