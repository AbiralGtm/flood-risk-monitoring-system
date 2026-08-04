"""
10_merge_and_finalize_training_data.py

PURPOSE
-------
Concatenates your historical_grid_training_rows.csv (1993-2020
reconstruction, grid-level, from step 09) with your real
ml_training_dataset.csv (2021-2026 observed hourly data, from step 03),
and recomputes risk_class/flood_label thresholds over the COMBINED
dataset -- not by reusing 03's original thresholds (which were computed
on observed-only data and don't reflect the full distribution once
historical rows are added) and not by computing historical thresholds
separately (which would create two inconsistent labeling scales in one
model).

Also adds an explicit data_source column across BOTH row sets (not just
the historical ones), so you can directly check afterward whether
XGBoost's feature importance leans on data_source -- per the earlier
research report's recommendation, that's your signal for whether to
split into two model tiers instead of merging into one.

INPUT
------
ml_training_dataset.csv             (from 03 -- real 2021-2026 data)
historical_grid_training_rows.csv   (from 09 -- reconstructed 1993-2020 data)

OUTPUT
------
ml_training_dataset_v2.csv
  Combined rows, with:
    data_source          : "observed_hourly" or "reanalysis_daily"
    risk_class            : RECOMPUTED over the combined final_risk_score
                             distribution (70th/90th/98th percentile,
                             same method as 03)
    flood_label            : recomputed to match
    risk_class_v1_observed_only : your ORIGINAL 03 risk_class, preserved
                             for comparison (NaN for historical rows,
                             which didn't have one)
"""

import sys
import numpy as np
import pandas as pd

OBSERVED_CSV = "ml_training_dataset.csv"
HISTORICAL_CSV = "historical_grid_training_rows.csv"
OUT_CSV = "ml_training_dataset_v2.csv"


def main():
    try:
        observed = pd.read_csv(OBSERVED_CSV, parse_dates=["event_time"])
    except FileNotFoundError:
        print(f"MISSING INPUT: {OBSERVED_CSV}")
        sys.exit(1)
    try:
        historical = pd.read_csv(HISTORICAL_CSV, parse_dates=["event_time"])
    except FileNotFoundError:
        print(f"MISSING INPUT: {HISTORICAL_CSV}")
        sys.exit(1)

    print(f"Loaded {len(observed):,} observed rows (2021-2026), "
          f"{len(historical):,} historical rows (1993-2020).")

    observed = observed.copy()
    observed["data_source"] = observed.get("data_source", "observed_hourly")
    observed["data_source"] = observed["data_source"].fillna("observed_hourly")
    observed["risk_class_v1_observed_only"] = observed["risk_class"]

    historical = historical.copy()
    if "data_source" not in historical.columns:
        historical["data_source"] = "reanalysis_daily"
    historical["risk_class_v1_observed_only"] = np.nan

    # Align columns -- keep the union, filling missing with NaN, so a
    # column present in one file but not the other doesn't silently drop
    # rows or misalign during concat.
    all_cols = sorted(set(observed.columns) | set(historical.columns))
    for col in all_cols:
        if col not in observed.columns:
            observed[col] = np.nan
        if col not in historical.columns:
            historical[col] = np.nan

    combined = pd.concat([observed[all_cols], historical[all_cols]], ignore_index=True)
    print(f"Combined: {len(combined):,} total rows")
    print(f"  data_source breakdown: {combined['data_source'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # FIX: compute risk_class thresholds from OBSERVED DATA ONLY, then
    # apply that SAME fixed boundary to classify every row (both eras).
    #
    # The original version of this script recalibrated thresholds on the
    # COMBINED distribution -- this silently redefines what "Danger"
    # means every time more historical data is added, since the combined
    # distribution skews differently (historical rows trend slightly
    # higher-risk on average). Confirmed via real training results: this
    # pushed the Danger threshold from [0.3039, 0.4202, 0.5827]
    # (observed-only, original baseline) up to [0.3362, 0.4725, 0.6143]
    # (combined), which dropped test-set Danger support from 809 to 481
    # rows and crashed XGBoost's Danger precision from 0.892 to 0.518 --
    # not because the model got worse, but because the TARGET DEFINITION
    # shifted underneath it. Locking thresholds to observed data keeps
    # "Danger" meaning a stable, consistent severity level regardless of
    # how much historical data you integrate.
    # ------------------------------------------------------------------
    before_n = len(combined)
    combined = combined.dropna(subset=["final_risk_score"])
    if len(combined) < before_n:
        print(f"WARNING: dropped {before_n - len(combined)} rows with missing "
              f"final_risk_score before threshold calculation.")

    observed_only_scores = combined.loc[
        combined["data_source"] == "observed_hourly", "final_risk_score"
    ]
    if len(observed_only_scores) == 0:
        print("WARNING: no observed_hourly rows found -- falling back to combined "
              "distribution for thresholds (this will redefine Danger, see note above).")
        observed_only_scores = combined["final_risk_score"]

    breaks = observed_only_scores.quantile([0.70, 0.90, 0.98]).values
    print(f"\nRisk class breaks (70th/90th/98th percentile, OBSERVED DATA ONLY -- "
          f"applied uniformly to both eras): {breaks.round(4)}")

    def classify_risk(s):
        if s <= breaks[0]: return "Low"
        if s <= breaks[1]: return "Medium"
        if s <= breaks[2]: return "High"
        return "Danger"

    combined["risk_class"] = combined["final_risk_score"].apply(classify_risk)
    combined["flood_label"] = combined["risk_class"].isin(["High", "Danger"]).astype(int)

    combined.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(combined):,} rows to {OUT_CSV}")
    print("\nrisk_class distribution (combined):")
    print(combined["risk_class"].value_counts())
    print(f"\nflood_label=1 fraction: {combined['flood_label'].mean():.3f}")

    print("\nrisk_class distribution BY data_source (check these aren't wildly")
    print("different -- if reanalysis_daily rows cluster heavily into one class,")
    print("that's a sign the reconstruction and observed data aren't on comparable")
    print("scales, and XGBoost may learn to key off data_source as a shortcut):")
    print(combined.groupby("data_source")["risk_class"].value_counts(normalize=True).round(3))

    # ------------------------------------------------------------------
    # Compare against the original observed-only labels, for rows that
    # have both -- how much did adding historical data shift the
    # thresholds for the SAME observed rows?
    # ------------------------------------------------------------------
    has_both = combined.dropna(subset=["risk_class_v1_observed_only"])
    if len(has_both):
        changed = (has_both["risk_class"] != has_both["risk_class_v1_observed_only"]).sum()
        print(f"\nOf {len(has_both):,} originally-observed rows, {changed:,} "
              f"({100*changed/len(has_both):.1f}%) changed risk_class after "
              f"recalibrating thresholds on the combined dataset.")
        if changed / len(has_both) > 0.15:
            print("NOTE: this is a meaningful shift -- worth re-running the calm-condition")
            print("false-positive check (from 03) against this new file before retraining,")
            print("to confirm the combined thresholds haven't reintroduced false positives.")

    print("\n--- SELF-VERIFICATION ---")
    checks_passed = True
    if combined["final_risk_score"].between(0, 1).all():
        print("PASS: final_risk_score within [0,1]")
    else:
        print("FAIL: final_risk_score out of range")
        checks_passed = False

    for split_name, split_df in [
        ("observed_hourly", combined[combined["data_source"] == "observed_hourly"]),
        ("reanalysis_daily", combined[combined["data_source"] == "reanalysis_daily"]),
    ]:
        n_danger = (split_df["risk_class"] == "Danger").sum()
        print(f"  Danger rows in {split_name}: {n_danger:,} / {len(split_df):,}")
        if n_danger == 0 and len(split_df) > 0:
            print(f"    WARNING: zero Danger rows in {split_name} -- check this is expected")

    print(f"\nOverall: {'PASS' if checks_passed else 'FAIL -- review warnings above'}")
    print(f"\nNEXT STEP: re-run the calm-condition check (from 03) against {OUT_CSV},")
    print("then run 04_train_models.py on this new file (update IN_CSV to point here)")
    print("and check feature importance for whether the model leans on data_source.")


if __name__ == "__main__":
    main()