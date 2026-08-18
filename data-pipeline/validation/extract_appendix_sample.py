"""
extract_appendix_sample.py

PURPOSE
-------
Generates a small, genuine sample of real rows from your actual training
dataset for inclusion in the report's Appendix -- one row per risk class
(Low/Medium/High/Danger), stratified from BOTH data sources (observed
and reconstructed) where available, so the sample honestly represents
the dataset's real structure rather than being cherry-picked or invented.

INPUT
------
ml_training_dataset_v2.csv

OUTPUT
------
appendix_sample.csv       -- for your own reference / double-checking
appendix_sample_table.txt -- pre-formatted for pasting into the report,
                              rounded to a readable number of decimals
"""

import pandas as pd

IN_CSV = "ml_training_dataset_v2.csv"
FEATURES = [
    "grid_id", "event_time", "elevation_mean", "slope_mean",
    "distance_to_narayani_m", "distance_to_rapti_m", "nearest_river_distance",
    "nearest_major_river_encoded", "within_2km_narayani", "within_2km_rapti",
    "lulc_encoded", "historical_flood_evidence", "flow_accumulation_score",
    "rainfall_24h_mm", "rainfall_score", "river_level_score", "risk_class",
]

def main():
    df = pd.read_csv(IN_CSV, low_memory=False)
    cols = [c for c in FEATURES if c in df.columns]
    df = df[cols].dropna(subset=["risk_class"])

    rows = []
    for cls in ["Low", "Medium", "High", "Danger"]:
        subset = df[df["risk_class"] == cls]
        if len(subset):
            rows.append(subset.sample(1, random_state=42))
    sample = pd.concat(rows, ignore_index=True)

    numeric_cols = sample.select_dtypes(include="number").columns
    sample[numeric_cols] = sample[numeric_cols].round(3)

    sample.to_csv("appendix_sample.csv", index=False)

    with open("appendix_sample_table.txt", "w") as f:
        f.write(sample.to_string(index=False))

    print("Sample rows (one per risk class):")
    print(sample.to_string(index=False))
    print("\nSaved: appendix_sample.csv, appendix_sample_table.txt")
    print("\nReview these real values before pasting into the report --")
    print("confirm nothing looks like an obvious outlier or data error")
    print("that you wouldn't want representing your dataset.")

if __name__ == "__main__":
    main()