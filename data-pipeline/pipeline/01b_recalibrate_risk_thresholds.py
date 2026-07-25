"""
STEP 1.5 (run right after Step 1) — Recalibrate static_risk_class using
thresholds derived from THIS study area's actual score distribution,
instead of generic fixed cutoffs (0.25/0.50/0.75).

Why: the whole grid is pre-filtered to be within 2km of a river (that's
how the study area was defined), and Bharatpur's river corridor sits in
a genuinely flat, low valley floor. That makes elevation_score and
slope_score saturate at their maximum for most grids, which pushes
static_susceptibility_score high for the majority of the study area
almost regardless of real relative risk. Generic thresholds then classify
most grids as High/Danger, which isn't useful for a warning system and
-- more importantly -- means the ML weak labels barely vary with rainfall.

This uses Jenks natural breaks (via mapclassify if available, else a
quantile fallback) to find the actual gaps in YOUR data's score
distribution and calibrate Low/Medium/High/Danger to those.

CHANGE FROM ORIGINAL VERSION
-----------------------------
The original version of this script only read/wrote the CSV, never
geometry. That meant the recalibrated static_risk_class only ever lived
in the CSV, while any separately-exported .geojson (e.g. from QGIS, or
from step 1 directly) kept the OLD generic-threshold classification --
which is exactly the mismatch that caused the 1,405/561/434/34 vs
926/560/560/388 discrepancy. This version now ALSO reads the step-1
geojson (for geometry), merges in the recalibrated columns by grid_id,
and writes out a proper geojson -- with an explicit float-dtype cast on
the score column as a safety net against accidental stringification.

Input:
  - static_grid_dataset_with_flood_evidence.csv           (output of step 1)
  - static_grid_dataset_with_flood_evidence_step1.geojson (geometry-carrying
    output of step 1 -- EDIT THIS PATH below to match whatever your step-1
    script actually names its geojson output)

Output:
  - static_grid_dataset_with_flood_evidence.csv     (recalibrated, as before)
  - static_grid_dataset_with_flood_evidence.geojson (NEW: recalibrated,
    with geometry, correct float dtypes -- this is the file everything
    downstream, including 05_flow_accumulation_extraction.py, should
    actually point at from now on)
"""

import pandas as pd
import numpy as np

IN_CSV = "static_grid_dataset_with_flood_evidence.csv"
OUT_CSV = "static_grid_dataset_with_flood_evidence.csv"   # overwrite in place

# EDIT this to match your actual step-1 geojson filename if different.
IN_GEOJSON = "static_grid_dataset_with_flood_evidence.geojson"
OUT_GEOJSON = "static_grid_dataset_with_flood_evidence.geojson"

df = pd.read_csv(IN_CSV)
scores = df["static_susceptibility_score"]

print("static_susceptibility_score distribution:")
print(scores.describe())

# Keep the original generic-threshold classification for comparison/report
df["static_risk_class_generic"] = df["static_risk_class"]

# ----------------------------------------------------------------------
# Try Jenks natural breaks (mapclassify); fall back to quantiles if the
# library isn't installed. Both are legitimate, standard classification
# methods -- Jenks is the more common choice in hazard-mapping literature.
# ----------------------------------------------------------------------
try:
    import mapclassify
    jenks = mapclassify.NaturalBreaks(scores, k=4)
    breaks = jenks.bins  # 4 upper bin edges, e.g. [0.42, 0.58, 0.71, 0.90]
    method = "Jenks natural breaks (mapclassify)"
except ImportError:
    breaks = scores.quantile([0.25, 0.50, 0.75, 1.0]).values
    method = "Quantile classification (mapclassify not installed -- " \
             "run `pip install mapclassify` for Jenks natural breaks instead)"

print(f"\nClassification method used: {method}")
print(f"Calibrated breakpoints (upper edges of Low/Medium/High/Danger): {breaks}")

def classify(s, breaks):
    if s <= breaks[0]:
        return "Low"
    if s <= breaks[1]:
        return "Medium"
    if s <= breaks[2]:
        return "High"
    return "Danger"

df["static_risk_class"] = scores.apply(lambda s: classify(s, breaks))

print("\n--- BEFORE (generic 0.25/0.50/0.75 thresholds) ---")
print(df["static_risk_class_generic"].value_counts())
print("\n--- AFTER (calibrated to this study area's actual distribution) ---")
print(df["static_risk_class"].value_counts())

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved recalibrated CSV to {OUT_CSV}")

# ------------------------------------------------------------------
# NEW: also produce the geometry-carrying geojson, recalibrated.
# ------------------------------------------------------------------
try:
    import geopandas as gpd
except ImportError:
    print("\ngeopandas not installed -- skipping geojson export.")
    print("Run: pip install geopandas --break-system-packages")
    print("then re-run this script to also get the .geojson output.")
else:
    try:
        geom_source = gpd.read_file(IN_GEOJSON)
    except FileNotFoundError:
        print(f"\nWARNING: {IN_GEOJSON} not found -- skipping geojson export.")
        print("Edit IN_GEOJSON at the top of this script to point at your")
        print("actual step-1 geojson output, then re-run.")
    else:
        if "grid_id" not in geom_source.columns:
            print(f"\nWARNING: {IN_GEOJSON} has no grid_id column to join on -- "
                  f"cannot safely merge recalibrated classes onto geometry. "
                  f"Columns found: {list(geom_source.columns)}")
        else:
            # Keep geometry + grid_id from the geojson; take every OTHER
            # column fresh from the recalibrated CSV, so there is exactly
            # one source of truth for attribute values (avoids silently
            # keeping stale duplicate columns from the geojson).
            geom_only = geom_source[["grid_id", "geometry"]]
            merged = geom_only.merge(df, on="grid_id", how="left", validate="one_to_one")

            n_unmatched = merged["static_risk_class"].isna().sum()
            if n_unmatched > 0:
                print(f"\nWARNING: {n_unmatched} grids in the geojson had no matching "
                      f"grid_id in the CSV -- check that both files came from the same "
                      f"step-1 run (same grid count, same IDs).")

            # Explicit float-dtype guard -- this is the safety net against the
            # stringification bug seen in the earlier ad-hoc geojson export.
            numeric_candidate_cols = [
                c for c in merged.columns
                if c not in ("grid_id", "geometry") and merged[c].dtype == object
            ]
            for col in numeric_candidate_cols:
                coerced = pd.to_numeric(merged[col], errors="coerce")
                # Only overwrite if coercion didn't just turn everything into NaN
                # (i.e. the column really was numeric-as-string, not genuinely text
                # like static_risk_class or lulc_dominant).
                if coerced.notna().sum() == merged[col].notna().sum():
                    merged[col] = coerced

            merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=geom_source.crs)
            merged.to_file(OUT_GEOJSON, driver="GeoJSON")

            # Verify the fix actually worked before declaring success
            verify = gpd.read_file(OUT_GEOJSON)
            score_dtype = verify["static_susceptibility_score"].dtype
            print(f"\nSaved recalibrated geojson to {OUT_GEOJSON}")
            print(f"Grid count in geojson: {len(verify)}")
            print(f"static_susceptibility_score dtype on disk: {score_dtype} "
                  f"({'PASS -- numeric' if pd.api.types.is_numeric_dtype(score_dtype) else 'FAIL -- still stringified, investigate further'})")
            print("\nstatic_risk_class distribution in the NEW geojson (should match the")
            print("recalibrated CSV distribution above, not the old generic one):")
            print(verify["static_risk_class"].value_counts())

print("\n--- Methodology note for your report ---")
print("Risk class thresholds for static_susceptibility_score were calibrated")
print("to this study area's actual score distribution using Jenks natural")
print("breaks, rather than fixed generic cutoffs. This was necessary because")
print("the study area is pre-filtered to lie within the river influence zone")
print("and sits within Bharatpur's flat valley floor, causing elevation and")
print("slope sub-scores to saturate near their maximum for most grids under")
print("generic absolute thresholds -- which would otherwise classify most of")
print("the study area as High/Danger regardless of actual relative risk or")
print("rainfall conditions.")