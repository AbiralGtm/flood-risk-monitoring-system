"""
Run this against your ACTUAL static_dataset_final.geojson (the one your
backend/training pipeline currently uses) to see exactly which grids are
incomplete and why -- before deciding whether this is a version mismatch
or a genuine data gap.
"""

import json
import pandas as pd

with open("static_dataset_final.geojson") as f:
    gj = json.load(f)

rows = [feat["properties"] for feat in gj["features"]]
df = pd.DataFrame(rows)

print(f"Total grids: {len(df)}")

missing_elev = df["elevation_mean"].isna()
missing_slope = df["slope_mean"].isna()
missing_any = missing_elev | missing_slope

print(f"Missing elevation: {missing_elev.sum()}")
print(f"Missing slope: {missing_slope.sum()}")
print(f"Missing either: {missing_any.sum()}")

incomplete = df[missing_any]
print(f"\narea_km2 stats for incomplete grids (small = edge slivers):")
print(incomplete["area_km2"].describe())

print(f"\nHow many incomplete grids are ALSO small (<0.01 km2, likely edge slivers):")
print((incomplete["area_km2"] < 0.01).sum(), "out of", len(incomplete))

print(f"\nHow many incomplete grids are NOT small (genuine gap, not just edge slivers):")
genuine_gap = incomplete[incomplete["area_km2"] >= 0.01]
print(len(genuine_gap))
if len(genuine_gap):
    print(genuine_gap[["grid_id", "area_km2", "centroid_lat", "centroid_lon"]].to_string(index=False))
    print("\n^ These are NOT edge slivers -- worth checking their location on")
    print("a map against your DEM/slope raster coverage. If they're clustered")
    print("in one area, your raster may not fully cover the study boundary there.")
