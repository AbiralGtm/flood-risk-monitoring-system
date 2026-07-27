"""
SAR post-processing — after exporting bharatpur_sar_flood_2024_09_28.geojson
from Google Earth Engine (see sar_flood_mapping_gee_script.js), intersect it
with your grid to find which grids were actually observed as flooded.

Output: sar_flooded_grids_2024-09-28.csv (grid_id list) -- this is exactly
the file 03_build_ml_training_dataset.py's commented-out SAR override block
expects. Once you run this, uncomment that block and point event_id at
whichever event_id in your selected_events.csv corresponds to Sept 28, 2024.

Inputs:
  - static_grid_dataset_with_flood_evidence.geojson  (your grid, has real
    polygon geometry)
  - bharatpur_sar_flood_2024_09_28.geojson            (from GEE export)

Output:
  - sar_flooded_grids_2024-09-28.csv
"""

import geopandas as gpd
import pandas as pd

GRID_GEOJSON = "static_grid_dataset_with_flood_evidence.geojson"
SAR_FLOOD_GEOJSON = "bharatpur_sar_flood_2024-09-28.geojson"
OUT_CSV = "sar_flooded_grids_2024-09-28.csv"

CRS_METRIC = "EPSG:32645"

grid = gpd.read_file(GRID_GEOJSON).to_crs(CRS_METRIC)
sar_flood = gpd.read_file(SAR_FLOOD_GEOJSON).to_crs(CRS_METRIC)

# Dissolve all flood polygons into one, for a clean single intersection test
sar_flood_dissolved = sar_flood.dissolve()

# A grid counts as "flooded" if the SAR extent overlaps ANY meaningful
# portion of it -- using intersects (not "within") since even partial
# inundation of a grid cell is meaningful for a 250m grid.
joined = gpd.sjoin(
    grid[["grid_id", "geometry"]],
    sar_flood_dissolved[["geometry"]],
    predicate="intersects",
    how="inner",
)

flooded_grid_ids = joined["grid_id"].unique()
print(f"Grids intersecting SAR-detected flood extent: {len(flooded_grid_ids)} "
      f"out of {len(grid)} ({100*len(flooded_grid_ids)/len(grid):.1f}%)")

pd.DataFrame({"grid_id": flooded_grid_ids}).to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")

print("\n--- Next step ---")
print("In 03_build_ml_training_dataset.py, uncomment the SAR override block")
print("and set sar_event_id to whichever event_id in selected_events.csv")
print("corresponds to 2024-09-28 (check via: pd.read_csv('selected_events.csv')")
print("and find the row where event_time is closest to that date).")
