"""
check_ml_training_dataset_v2.py

PURPOSE
-------
Verifies ml_training_dataset_v2.csv is actually safe to train on, before
you spend Kaggle/local compute time on it. Checks the specific issues
this pipeline has already hit:
  1. The false-zero problem: rainfall_1h/3h/6h/12h_mm get set to a
     literal 0.0 (not NaN) for historical rows during IDW blending,
     since CHIRPS only provides daily data. This is now FIXED by
     excluding those columns from FEATURES in 04_train_models.py --
     this script confirms the problem is real in your actual file,
     and that excluding them is the right call.
  2. Row-count sanity (catches a repeat of the earlier 15GB/28x-
     imbalance bug if it ever recurs).
  3. NaN coverage in the columns 04 actually needs.
  4. risk_class / data_source distribution sanity.
"""

import sys
import pandas as pd

IN_CSV = "ml_training_dataset_v2.csv"

FEATURES_TO_CHECK = [
    "elevation_mean", "slope_mean",
    "distance_to_narayani_m", "distance_to_rapti_m", "nearest_river_distance",
    "nearest_major_river_encoded",
    "within_2km_narayani", "within_2km_rapti",
    "lulc_encoded", "historical_flood_evidence",
    "flow_accumulation_score",
    "rainfall_1h_mm", "rainfall_3h_mm", "rainfall_6h_mm",
    "rainfall_12h_mm", "rainfall_24h_mm", "rainfall_score",
    "water_level_m", "river_level_score",
]
SUBDAILY_RAIN_COLS = ["rainfall_1h_mm", "rainfall_3h_mm", "rainfall_6h_mm", "rainfall_12h_mm"]


def main():
    try:
        df = pd.read_csv(IN_CSV, low_memory=False)
    except FileNotFoundError:
        print(f"MISSING: {IN_CSV}")
        sys.exit(1)

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.\n")

    # --- 1. Scale sanity ---
    print("=== SCALE CHECK ===")
    if "data_source" in df.columns:
        counts = df["data_source"].value_counts()
        print(counts)
        if "reanalysis_daily" in counts and "observed_hourly" in counts:
            ratio = counts.get("reanalysis_daily", 0) / max(counts.get("observed_hourly", 1), 1)
            print(f"reanalysis_daily / observed_hourly ratio: {ratio:.2f}x")
            if ratio > 5:
                print("WARNING: historical data still heavily outweighs observed data -- "
                      "check you're using the SAMPLED historical file, not the full one.")
            else:
                print("OK: ratio is reasonable, not the 28x imbalance seen earlier.")
    else:
        print("No data_source column found.")
    print()

    # --- 2. THE FALSE-ZERO CHECK (the critical one) ---
    print("=== FALSE-ZERO CHECK (sub-daily rainfall columns) ===")
    if "data_source" in df.columns:
        for col in SUBDAILY_RAIN_COLS:
            if col not in df.columns:
                print(f"  {col}: NOT PRESENT in file")
                continue
            hist_rows = df[df["data_source"] == "reanalysis_daily"]
            pct_exact_zero = (hist_rows[col] == 0).mean() * 100 if len(hist_rows) else float("nan")
            print(f"  {col}: {pct_exact_zero:.1f}% of reanalysis_daily rows are EXACTLY 0.0")
        print()
        print("If these percentages are near 100%, that CONFIRMS the false-zero problem --")
        print("these columns must be excluded from FEATURES in 04_train_models.py.")
        print("(rainfall_24h_mm and rainfall_score are NOT affected -- CHIRPS provides")
        print("real values for those on every historical row.)")
    else:
        print("Cannot check without data_source column.")
    print()

    # --- 3. NaN coverage in features 04 actually needs ---
    print("=== NaN COVERAGE (features 04_train_models.py uses) ===")
    for col in FEATURES_TO_CHECK:
        if col not in df.columns:
            print(f"  {col}: MISSING FROM FILE")
            continue
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            print(f"  {col}: {n_nan:,} NaN ({100*n_nan/len(df):.1f}%)")
    print("  (columns not listed above have zero NaN)")
    print()

    # --- 4. Distribution sanity ---
    print("=== risk_class DISTRIBUTION ===")
    if "risk_class" in df.columns:
        print(df["risk_class"].value_counts())
        print(f"\nflood_label=1 fraction: {df['flood_label'].mean():.3f}")
    print()

    if "data_source" in df.columns and "risk_class" in df.columns:
        print("=== risk_class BY data_source ===")
        print(df.groupby("data_source")["risk_class"].value_counts(normalize=True).round(3))

    print("\n=== RECOMMENDATION ===")
    print("If the false-zero check above shows ~100% zeros for reanalysis_daily rows in")
    print("rainfall_1h/3h/6h/12h_mm, use the CORRECTED 04_train_models.py (with those four")
    print("columns removed from FEATURES) -- NOT the version with them still included.")


if __name__ == "__main__":
    main()
