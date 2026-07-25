"""
verify_grid.py -- Run FIRST, before the pipeline. Validates your corrected
QGIS-exported static_dataset_final.geojson and prints exactly which cells
need visual checking in QGIS.

Usage: python verify_grid.py
"""

import json
import pandas as pd

GEOJSON = "static_dataset_final.geojson"
AREA_MIN = 0.005

EXPECTED_COLS = [
    "grid_id", "area_km2", "centroid_lon", "centroid_lat",
    "distance_to_narayani_m", "distance_to_rapti_m", "nearest_river_distance",
    "nearest_major_river", "within_2km_narayani", "within_2km_rapti",
    "river_distance_score", "elevation_mean", "elevation_min", "elevation_max",
    "elevation_score", "slope_mean", "slope_score",
    "lulc_dominant", "lulc_encoded", "lulc_score",
]

with open(GEOJSON) as f:
    gj = json.load(f)

rows = [feat["properties"] for feat in gj["features"]]
df = pd.DataFrame(rows)

print(f"=== {GEOJSON} ===")
print(f"Features: {len(df)}")

# 1. Column check
missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
if missing_cols:
    print(f"\n*** MISSING EXPECTED COLUMNS: {missing_cols} ***")
    print("Fix these in QGIS before running the pipeline.")
else:
    print("All expected columns present. ✓")

# 2. Duplicates
dupes = df["grid_id"].duplicated().sum()
print(f"Duplicate grid_id: {dupes} {'✓' if dupes == 0 else '*** FIX THIS ***'}")

# 3. Slivers and missing values
slivers = (df["area_km2"] < AREA_MIN).sum()
clean = df[df["area_km2"] >= AREA_MIN]
print(f"\nSlivers (area < {AREA_MIN} km2): {slivers} -> {len(clean)} clean grids")
print(f"Missing elevation after sliver removal: {clean['elevation_mean'].isna().sum()}")
print(f"Missing slope after sliver removal: {clean['slope_mean'].isna().sum()}")

# 4. River split
print(f"\nRiver split:")
print(clean["nearest_major_river"].value_counts().to_string())

# 5. Zero/near-zero river distance cells -- the ones to check visually
suspicious = clean[clean["nearest_river_distance"] < 5][
    ["grid_id", "centroid_lat", "centroid_lon", "nearest_river_distance", "nearest_major_river"]
]
print(f"\nCells with nearest_river_distance < 5m: {len(suspicious)}")
if len(suspicious):
    print("CHECK THESE VISUALLY in QGIS against satellite imagery --")
    print("bankside land is fine; open water means the erase step missed a spot:")
    print(suspicious.to_string(index=False))

# 6. Score field sanity
print("\nScore field values (should be clean tiers):")
for col in ["river_distance_score", "elevation_score", "slope_score", "lulc_score"]:
    if col in clean.columns:
        print(f"  {col}: {sorted(clean[col].dropna().unique())}")

# 7. LULC water-dominant cells -- should be rare/absent now that the
#    river was erased; a few small ponds are fine, many cells is not
water_cells = (clean["lulc_dominant"] == "Water").sum()
print(f"\nGrids with Water as dominant LULC: {water_cells}")
if water_cells > 20:
    print("  ^ Higher than expected for a river-erased grid -- worth checking")
    print("    whether these are ponds/wetlands (fine) or river remnants (not fine).")

print("\n=== VERDICT ===")
ok = (dupes == 0 and not missing_cols
      and clean["elevation_mean"].isna().sum() == 0)
print("READY for pipeline ✓" if ok else "Fix the flagged issues first, then re-run this check.")
