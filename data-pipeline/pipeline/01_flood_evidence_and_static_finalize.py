"""
STEP 1 — Clean sliver grids, integrate flood evidence using REAL POLYGON
GEOMETRY (your static_dataset_final.geojson has actual MultiPolygon grid
cells, already in EPSG:32645 -- no reprojection needed for the grid side),
recompute static_susceptibility_score and static_risk_class, export.

Uses "intersects" (not "within") for the evidence join: a flood evidence
buffer overlapping ANY part of a grid cell counts as touching that grid --
this is more accurate than a centroid-only join now that you have real
polygons, since a buffer can clip a grid's edge without containing its
center.

Inputs:
  - static_dataset_final.geojson   (as uploaded -- no changes needed)
  - flood_evidence_bharatpur.csv

Output:
  - static_grid_dataset_with_flood_evidence.geojson  (keeps geometry, for
    your frontend map layer)
  - static_grid_dataset_with_flood_evidence.csv       (attributes only,
    for the ML pipeline steps that follow)
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

GRID_GEOJSON = "static_dataset_final.geojson"
FLOOD_EVIDENCE_CSV = "flood_evidence_bharatpur.csv"

OUT_GEOJSON = "static_grid_dataset_with_flood_evidence.geojson"
OUT_CSV = "static_grid_dataset_with_flood_evidence.csv"

CRS_METRIC = "EPSG:32645"   # matches the uploaded file already
CRS_GEO = "EPSG:4326"       # flood evidence lat/lon comes in as WGS84

AREA_KM2_MIN = 0.005
BUFFER_BY_ACCURACY = {"exact": 500, "settlement": 650, "ward": None}
CONFIDENCE_WEIGHTS = {"High": 1.0, "Medium": 0.6, "Low": 0.3}

# ----------------------------------------------------------------------
# 1. Load grid (already has real polygon geometry + correct CRS)
# ----------------------------------------------------------------------
grid = gpd.read_file(GRID_GEOJSON)
if grid.crs is None:
    grid = grid.set_crs(CRS_METRIC)
grid = grid.to_crs(CRS_METRIC)

before = len(grid)
grid = grid[grid["area_km2"] >= AREA_KM2_MIN].copy()
print(f"Removed {before - len(grid)} sliver grids (area_km2 < {AREA_KM2_MIN}); {len(grid)} grids remain.")
assert grid["grid_id"].is_unique, "Duplicate grid_id -- investigate before continuing."

# A small number of edge-clipped cells can still have NoData slope/elevation
# even above the area threshold (DEM/slope raster edge effects at the
# corrected study boundary). Same rationale as sliver removal: unreliable
# raster stats on tiny/clipped cells -- drop them rather than crash.
missing_slope = grid["slope_mean"].isna().sum()
missing_elev = grid["elevation_mean"].isna().sum()
if missing_slope or missing_elev:
    print(f"Dropping {missing_slope} grid(s) with missing slope and "
          f"{missing_elev} with missing elevation (edge-clip NoData, same "
          f"rationale as sliver removal).")
    grid = grid.dropna(subset=["slope_mean", "elevation_mean"]).copy()
    print(f"{len(grid)} grids remain after this additional cleaning step.")

# ----------------------------------------------------------------------
# 2. Load flood evidence, build points, reproject, buffer by accuracy
# ----------------------------------------------------------------------
ev = pd.read_csv(FLOOD_EVIDENCE_CSV)
ev = ev.dropna(subset=["latitude", "longitude"]).copy()
ev_gdf = gpd.GeoDataFrame(
    ev,
    geometry=[Point(xy) for xy in zip(ev["longitude"], ev["latitude"])],
    crs=CRS_GEO,
).to_crs(CRS_METRIC)

def buffer_radius(row):
    acc = str(row["spatial_accuracy"]).strip().lower()
    return BUFFER_BY_ACCURACY.get(acc, BUFFER_BY_ACCURACY["settlement"])

ev_gdf["buffer_m"] = ev_gdf.apply(buffer_radius, axis=1)
usable_ev = ev_gdf[ev_gdf["buffer_m"].notna()].copy()
excluded = len(ev_gdf) - len(usable_ev)
if excluded:
    print(f"NOTE: {excluded} ward-only evidence rows excluded from point buffering.")

usable_ev["geometry"] = usable_ev.apply(lambda r: r.geometry.buffer(r["buffer_m"]), axis=1)
usable_ev["confidence_weight"] = usable_ev["confidence_level"].map(CONFIDENCE_WEIGHTS).fillna(0.3)

# ----------------------------------------------------------------------
# 3. Spatial join: grid POLYGONS that intersect evidence buffer POLYGONS
# ----------------------------------------------------------------------
joined = gpd.sjoin(
    grid[["grid_id", "geometry"]],
    usable_ev[["evidence_id", "confidence_weight", "geometry"]],
    predicate="intersects",
    how="left",
)

count_per_grid = (
    joined.dropna(subset=["evidence_id"]).groupby("grid_id")["evidence_id"]
    .nunique().rename("historical_flood_count")
)
weighted_per_grid = (
    joined.dropna(subset=["evidence_id"]).groupby("grid_id")["confidence_weight"]
    .sum().rename("history_weighted_sum")
)

grid = grid.merge(count_per_grid, on="grid_id", how="left")
grid = grid.merge(weighted_per_grid, on="grid_id", how="left")
grid["historical_flood_count"] = grid["historical_flood_count"].fillna(0).astype(int)
grid["history_weighted_sum"] = grid["history_weighted_sum"].fillna(0.0)
grid["historical_flood_evidence"] = (grid["historical_flood_count"] > 0).astype(int)

print(f"\nGrids touched by flood evidence: {grid['historical_flood_evidence'].sum()} out of {len(grid)}")

# ----------------------------------------------------------------------
# 4. history_score (count tier x confidence)
# ----------------------------------------------------------------------
def count_tier(c):
    if c >= 3: return 1.0
    if c == 2: return 0.8
    if c == 1: return 0.6
    return 0.0

grid["history_score_raw_tier"] = grid["historical_flood_count"].apply(count_tier)
grid["avg_confidence_weight"] = (
    grid["history_weighted_sum"] / grid["historical_flood_count"].replace(0, pd.NA)
).fillna(0.0)
grid["history_score"] = (
    grid["history_score_raw_tier"] * (0.5 + 0.5 * grid["avg_confidence_weight"])
).clip(0, 1)

# ----------------------------------------------------------------------
# 5. Recompute static_susceptibility_score and static_risk_class
# ----------------------------------------------------------------------
for col in ["river_distance_score", "elevation_score", "slope_score", "lulc_score"]:
    assert col in grid.columns, f"Missing expected column: {col}"

grid["static_susceptibility_score"] = (
    0.30 * grid["river_distance_score"]
    + 0.25 * grid["elevation_score"]
    + 0.20 * grid["slope_score"]
    + 0.15 * grid["lulc_score"]
    + 0.10 * grid["history_score"]
)

def risk_class(s):
    if s >= 0.75: return "Danger"
    if s >= 0.50: return "High"
    if s >= 0.25: return "Medium"
    return "Low"

grid["static_risk_class"] = grid["static_susceptibility_score"].apply(risk_class)

# ----------------------------------------------------------------------
# 6. Export both -- GeoJSON (with geometry, for the frontend map) and
#    CSV (attributes only, for the rest of the ML pipeline)
# ----------------------------------------------------------------------
grid.drop(columns=["history_score_raw_tier"]).to_file(OUT_GEOJSON, driver="GeoJSON")
grid.drop(columns=["geometry", "history_score_raw_tier"]).to_csv(OUT_CSV, index=False)

print(f"\nExported {len(grid)} grids to:\n  {OUT_GEOJSON}\n  {OUT_CSV}")
print("\nstatic_risk_class distribution:")
print(grid["static_risk_class"].value_counts())