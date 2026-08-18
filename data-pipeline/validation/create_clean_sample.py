"""
create_clean_sample.py

Creates a clean, representative sample of the full ML training dataset for
demonstration/appendix purposes -- at least 100,000 rows, keeping only the
columns actually used by the project (the 14 real model input features plus
identifying/label columns), and dropping the columns already established as
unused.

WHY THESE COLUMNS ARE DROPPED (matches the project's own documented decision,
see Step 13 of the technical build log):
  - rainfall_1h_mm, rainfall_3h_mm, rainfall_6h_mm, rainfall_12h_mm:
    silently zero across the entire historical CHIRPS reconstruction
    (CHIRPS only provides daily totals), excluded from the final feature set.
  - water_level_m (raw): 38% missing (concentrated on Rapti-side gauge gaps),
    redundant with the derived river_level_score once the fallback
    convention is applied.

WHY A RANDOM SAMPLE, NOT THE FIRST N ROWS:
  Taking the first 100,000 rows would likely just be the earliest dates in
  the dataset, which is not representative of the full time range or class
  balance. A random sample (with a fixed seed for reproducibility) avoids
  this.

INPUT:  ml_training_dataset_v2.csv
OUTPUT: ml_training_dataset_clean_sample.csv
"""

import pandas as pd
import numpy as np

IN_CSV = "ml_training_dataset_v2.csv"
OUT_CSV = "ml_training_dataset_clean_sample.csv"
SAMPLE_SIZE = 120_000  # comfortably over the requested 100,000
RANDOM_SEED = 42

# The 14 real model input features (established, final feature set)
MODEL_FEATURES = [
    "elevation_mean", "slope_mean", "distance_to_narayani_m", "distance_to_rapti_m",
    "nearest_river_distance", "nearest_major_river_encoded", "within_2km_narayani",
    "within_2km_rapti", "lulc_encoded", "historical_flood_evidence",
    "flow_accumulation_score", "rainfall_24h_mm", "rainfall_score", "river_level_score",
]

# Identifying / label columns worth keeping alongside the features, for context
CONTEXT_COLS = ["grid_id", "event_time", "data_source", "risk_class"]

# Columns explicitly excluded (kept here only as documentation of what's dropped)
EXCLUDED_COLS = ["rainfall_1h_mm", "rainfall_3h_mm", "rainfall_6h_mm",
                  "rainfall_12h_mm", "water_level_m"]


def main():
    print(f"Loading {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f"Full dataset: {len(df):,} rows, {len(df.columns)} columns")

    keep_cols = [c for c in CONTEXT_COLS + MODEL_FEATURES if c in df.columns]
    missing_expected = [c for c in CONTEXT_COLS + MODEL_FEATURES if c not in df.columns]
    if missing_expected:
        print(f"WARNING: expected columns not found in the source file: {missing_expected}")

    present_excluded = [c for c in EXCLUDED_COLS if c in df.columns]
    print(f"\nDropping {len(present_excluded)} unused columns: {present_excluded}")
    print(f"Keeping {len(keep_cols)} columns: {keep_cols}")

    df_clean = df[keep_cols].copy()

    if len(df_clean) < SAMPLE_SIZE:
        raise ValueError(
            f"Requested sample size ({SAMPLE_SIZE:,}) exceeds the full "
            f"dataset size ({len(df_clean):,}) -- reduce SAMPLE_SIZE."
        )

    sample = df_clean.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\nSample size: {len(sample):,} rows")
    print("\nrisk_class distribution in the sample:")
    print(sample["risk_class"].value_counts())
    print("\nrisk_class distribution in the FULL dataset (for comparison):")
    print(df_clean["risk_class"].value_counts(normalize=True).round(4))
    print("\nSample risk_class proportions (should roughly match full dataset above):")
    print(sample["risk_class"].value_counts(normalize=True).round(4))

    if "data_source" in sample.columns:
        print("\ndata_source distribution in the sample:")
        print(sample["data_source"].value_counts())

    n_nulls = sample.isnull().sum()
    n_nulls = n_nulls[n_nulls > 0]
    if len(n_nulls):
        print(f"\nColumns with missing values in the sample (expected -- e.g. river_level_score gaps):")
        print(n_nulls)

    sample.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV} ({len(sample):,} rows, {len(sample.columns)} columns)")


if __name__ == "__main__":
    main()