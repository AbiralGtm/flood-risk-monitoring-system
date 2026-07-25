"""
STEP 2 (run after Step 1) — Compute rainfall coverage for every grid
using Inverse Distance Weighting (IDW) across ALL 11 nearby stations,
not just the single nearest one. River-level assignment stays
nearest-single-gauge (river stage doesn't blend spatially the way
rainfall does -- standard hydrological practice).

Why IDW for rainfall now: with 11 stations spread roughly 1-32km around
Bharatpur (not just 2 central ones), a grid near Meghauli can now draw
more heavily from Meghauli/Riukhola than from central Bharatpur stations,
which is a real accuracy improvement over nearest-neighbor. With only
2-3 stations this wouldn't have been worth the complexity; with 11 it is.

Formula: weight_i = 1 / distance_i^IDW_POWER, normalized so weights for
each grid sum to 1 across all stations. Standard IDW, power=2 is the
conventional default.

UPDATE: GRID_GEOJSON now points at the output of 05_flow_accumulation_extraction.py
(static_dataset_with_flowacc.geojson) instead of step 1's output directly.
That file has everything step 1's output has, PLUS flow_accumulation_score
and a static_susceptibility_score/static_risk_class rebalanced to include it.
If you read step 1's output here instead, flow_accumulation never reaches
the training data no matter how well it validates on its own.

Inputs:
  - static_dataset_with_flowacc.geojson  (from step 05, includes flow accumulation)
  - rainfall_stations.csv   (11 Chitwan stations)
  - river_stations.csv      (Devghat, Rajaiya)

Outputs:
  - grid_rainfall_idw_weights.csv   (LONG format: grid_id, rainfall_station_id,
    weight -- one row per grid-station pair, weights sum to 1 per grid)
  - static_grid_dataset_with_stations.geojson / .csv   (grid + nearest
    river station only, since river stays single-nearest)
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

GRID_GEOJSON = "static_dataset_with_flowacc.geojson"  # was: static_grid_dataset_with_flood_evidence.geojson
RAINFALL_STATIONS_CSV = "rainfall_stations.csv"
RIVER_STATIONS_CSV = "river_stations.csv"

OUT_WEIGHTS_CSV = "grid_rainfall_idw_weights.csv"
OUT_GEOJSON = "static_grid_dataset_with_stations.geojson"
OUT_CSV = "static_grid_dataset_with_stations.csv"

CRS_METRIC = "EPSG:32645"
CRS_GEO = "EPSG:4326"

IDW_POWER = 2            # standard IDW exponent
MAX_STATIONS = 5         # use the 5 nearest stations per grid, not all 11 --
                          # keeps far stations (30km+) from adding noise/cost
                          # while still blending across several nearby ones

# ----------------------------------------------------------------------
grid = gpd.read_file(GRID_GEOJSON).to_crs(CRS_METRIC)

def load_stations(path):
    s = pd.read_csv(path)
    return gpd.GeoDataFrame(
        s, geometry=[Point(xy) for xy in zip(s["longitude"], s["latitude"])], crs=CRS_GEO
    ).to_crs(CRS_METRIC)

rainfall_stations = load_stations(RAINFALL_STATIONS_CSV)
river_stations_all = load_stations(RIVER_STATIONS_CSV)
if "active_in_pipeline" in river_stations_all.columns:
    river_stations = river_stations_all[river_stations_all["active_in_pipeline"] == True].copy()
    excluded = river_stations_all[river_stations_all["active_in_pipeline"] != True]
    if len(excluded):
        print(f"Excluding {len(excluded)} river station(s) from nearest-gauge assignment "
              f"(marked inactive -- see notes column): {list(excluded['station_name'])}")
else:
    river_stations = river_stations_all

# ----------------------------------------------------------------------
# 1. IDW weights for rainfall -- K nearest stations per grid
# ----------------------------------------------------------------------
rf_coords = list(zip(rainfall_stations.geometry.x, rainfall_stations.geometry.y))
rf_ids = rainfall_stations["station_id"].tolist()

weight_rows = []
for _, g in grid[["grid_id", "geometry"]].iterrows():
    gx, gy = g.geometry.centroid.x, g.geometry.centroid.y
    dists = [((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5 for sx, sy in rf_coords]
    pairs = sorted(zip(rf_ids, dists), key=lambda t: t[1])[:MAX_STATIONS]
    # avoid divide-by-zero if a grid centroid coincides exactly with a station
    inv = [1.0 / (d ** IDW_POWER) if d > 1 else 1.0 / (1 ** IDW_POWER) for _, d in pairs]
    total = sum(inv)
    for (sid, d), w in zip(pairs, inv):
        weight_rows.append({
            "grid_id": g["grid_id"],
            "rainfall_station_id": sid,
            "distance_m": d,
            "weight": w / total,
        })

weights_df = pd.DataFrame(weight_rows)
weights_df.to_csv(OUT_WEIGHTS_CSV, index=False)
print(f"Saved IDW weights: {len(weights_df)} rows "
      f"({grid.shape[0]} grids x up to {MAX_STATIONS} nearest stations each)")
print("\nHow often each station appears in a grid's nearest-K set:")
print(weights_df["rainfall_station_id"].value_counts())

# ----------------------------------------------------------------------
# 2. River level: nearest single gauge (unchanged logic)
# ----------------------------------------------------------------------
def assign_nearest(grid_gdf, stations_gdf, id_col, dist_col):
    nearest = gpd.sjoin_nearest(
        grid_gdf[["grid_id", "geometry"]],
        stations_gdf[["station_id", "geometry"]].rename(columns={"station_id": id_col}),
        distance_col=dist_col,
    )
    return nearest.drop_duplicates(subset="grid_id", keep="first")[["grid_id", id_col, dist_col]]

# ----------------------------------------------------------------------
# River station assignment by RIVER IDENTITY, not raw distance.
#
# Why: Devghat sits ~1km from Bharatpur while Rajaiya sits ~62km away.
# Straight-line nearest-gauge assignment would therefore pick Devghat for
# EVERY grid, including ones physically on the Rapti side -- Rajaiya can
# never geometrically "win" against a gauge that close, regardless of
# which river actually runs past a given grid. The correct rule is to
# assign each grid to the gauge on ITS river, using the nearest_major_river
# column already computed in your static GIS workflow (Narayani/Rapti),
# not distance-to-gauge.
# ----------------------------------------------------------------------
RIVER_TO_GAUGE = {"Narayani": "RV01", "Rapti": "RV02"}

if "nearest_major_river" in grid.columns:
    grid["river_station_id"] = grid["nearest_major_river"].map(RIVER_TO_GAUGE)
    # distance is still useful to report, even though it no longer drives assignment
    dist_lookup = assign_nearest(grid, river_stations, "_nearest_gauge_geometric", "dist_to_river_station_m")
    grid = grid.merge(dist_lookup[["grid_id", "dist_to_river_station_m"]], on="grid_id", how="left")
    print("Assigned river station by nearest_major_river identity (Narayani->RV01 Devghat, "
          "Rapti->RV02 Rajaiya), not raw geometric distance.")
else:
    river_assign = assign_nearest(grid, river_stations, "river_station_id", "dist_to_river_station_m")
    grid = grid.merge(river_assign, on="grid_id", how="left")
    print("WARNING: nearest_major_river column not found -- fell back to raw geometric "
          "nearest-gauge, which will likely assign every grid to Devghat. Check your input file.")

if {"warning_level_m", "danger_level_m"}.issubset(river_stations.columns):
    thresholds = river_stations.rename(columns={"station_id": "river_station_id"})[
        ["river_station_id", "warning_level_m", "danger_level_m"]
    ]
    grid = grid.merge(thresholds, on="river_station_id", how="left")

grid.to_file(OUT_GEOJSON, driver="GeoJSON")
grid.drop(columns="geometry").to_csv(OUT_CSV, index=False)

print(f"\nGrids per nearest river station:")
print(grid["river_station_id"].value_counts())
print(f"\nSaved: {OUT_WEIGHTS_CSV}, {OUT_GEOJSON}, {OUT_CSV}")
print("\n--- Methodology note for your report ---")
print("Rainfall was assigned to grids via Inverse Distance Weighting (IDW,")
print("power=2) across the 5 nearest of 11 active DHM stations in Chitwan")
print("district, rather than nearest-single-station assignment -- justified")
print("by the stations' genuine spatial spread (~1-32km around Bharatpur).")
print("River level was assigned by river identity (nearest_major_river:")
print("Narayani->Devghat, Rapti->Rajaiya), not by raw geometric nearest-gauge")
print("distance -- Rajaiya's 62km offset from Bharatpur would otherwise cause")
print("Devghat to be selected for every grid regardless of which river it")
print("actually borders.")
