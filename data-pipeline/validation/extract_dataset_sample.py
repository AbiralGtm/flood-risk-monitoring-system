"""
extract_dataset_sample.py

Selects 6 REAL rows from your actual training dataset for the report's
dataset-sample table: one row per risk class (Low, Medium, High, Danger),
plus 2 additional rows for variety. Keeps only the columns that matter for
this table: grid_id (for reference), the 14 real model input features, and
risk_class (the label -- shown for context, not a model input).

INPUT:  ml_training_dataset_v2.csv  (adjust IN_CSV below if your filename differs)
OUTPUT: dataset_sample_selected.csv  (6 rows, small, human-readable --
        check it yourself before generating the Word table)
"""

import pandas as pd

IN_CSV = "ml_training_dataset_v2.csv"   # adjust if your real filename differs
OUT_CSV = "dataset_sample_selected.csv"
RANDOM_SEED = 42

MODEL_FEATURES = [
    "elevation_mean", "slope_mean", "distance_to_narayani_m", "distance_to_rapti_m",
    "nearest_river_distance", "nearest_major_river_encoded", "within_2km_narayani",
    "within_2km_rapti", "lulc_encoded", "historical_flood_evidence",
    "flow_accumulation_score", "rainfall_24h_mm", "rainfall_score", "river_level_score",
]
KEEP_COLS = ["grid_id"] + MODEL_FEATURES + ["risk_class"]
RISK_ORDER = ["Low", "Medium", "High", "Danger"]


def main():
    print(f"Loading {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f"Full dataset: {len(df):,} rows")

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found: {missing}. "
                          f"Check your real column names and adjust the script.")

    df = df.dropna(subset=KEEP_COLS)

    # One real row per risk class, chosen deterministically (reproducible)
    picked_rows = []
    for cls in RISK_ORDER:
        subset = df[df["risk_class"] == cls]
        if len(subset) == 0:
            print(f"WARNING: no rows found for risk_class='{cls}', skipping")
            continue
        row = subset.sample(n=1, random_state=RANDOM_SEED)
        picked_rows.append(row)

    # 2 additional rows for variety, excluding already-picked grid_id+index combos
    already_picked_idx = pd.concat(picked_rows).index
    remaining = df.drop(index=already_picked_idx)
    extra = remaining.sample(n=min(2, len(remaining)), random_state=RANDOM_SEED)
    picked_rows.append(extra)

    sample = pd.concat(picked_rows)[KEEP_COLS].reset_index(drop=True)

    print(f"\nSelected {len(sample)} rows:")
    print(sample[["grid_id", "risk_class"]])

    sample.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print("Review this file, then run build_sample_table_docx.js to generate the Word table.")


if __name__ == "__main__":
    main()
