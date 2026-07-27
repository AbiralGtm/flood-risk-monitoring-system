"""
Run Google Earth Engine's Sentinel-1 flood detection from your local
Python environment -- same GEE cloud processing power (no need to wait
on HyP3, no local raster math needed), but scripted and runnable from
your laptop instead of the browser JS code editor.

ONE-TIME SETUP:
    pip install earthengine-api geemap

    Then authenticate once (opens a browser for Google login, saves a
    token locally so future runs don't re-prompt):
    earthengine authenticate

    If you get a "project required" error, you may also need a free
    Google Cloud project registered for Earth Engine at
    code.earthengine.google.com/register -- follow the prompt if asked.

WHAT THIS DOES (mirrors the JS version, run entirely from Python):
    1. Connects to GEE using your authenticated account
    2. Defines the study area (rectangle -- or point to your uploaded
       boundary asset if you have one)
    3. Loads Sentinel-1 VH imagery, before/after Sept 28 2024
    4. Computes flood extent via backscatter change detection
    5. Masks out permanent water (JRC Global Surface Water)
    6. Exports the flood polygon DIRECTLY to a local GeoJSON file --
       no manual Google Drive download step needed
"""

import ee
import geemap

# ----------------------------------------------------------------------
# 1. Authenticate and initialize
# ----------------------------------------------------------------------
# Newer earthengine-api versions require an explicit Google Cloud project.
# Replace with YOUR actual project ID from console.cloud.google.com
# (the Project ID, not the display name).
EE_PROJECT_ID = "winged-yeti-345514"

try:
    ee.Initialize(project=EE_PROJECT_ID)
except Exception:
    ee.Authenticate()  # opens browser once if not already authenticated
    ee.Initialize(project=EE_PROJECT_ID)

print("Connected to Earth Engine.")

# ----------------------------------------------------------------------
# 2. Study area
# ----------------------------------------------------------------------
# Option A: simple rectangle (works immediately, no asset upload needed)
aoi = ee.Geometry.Rectangle([84.15, 27.40, 84.75, 27.75])

# Option B: use your uploaded boundary asset instead, if you have one --
# uncomment and replace with your actual asset path:
# aoi = ee.FeatureCollection('projects/ee-yourusername/assets/bharatpur_boundary').geometry()

# ----------------------------------------------------------------------
# 3. Sentinel-1 collection, same logic as the JS version
# ----------------------------------------------------------------------
s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(aoi)
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
      .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
      .select('VH'))

before_col = s1.filterDate('2024-09-01', '2024-09-20')

# Sentinel-1B failed in Dec 2021 -- since then only S1A is operational
# over this region, meaning the REAL revisit cycle here is 12 days, not
# 6. Widening from the first (too-narrow) 3-day attempt to a 7-day
# window that should catch the next 12-day-cycle pass after mid-September.
after_col = s1.filterDate('2024-09-28', '2024-10-04')

print(f"Images in after-window: {after_col.size().getInfo()}")
print("Dates available:", after_col.aggregate_array('system:time_start').map(
    lambda t: ee.Date(t).format('YYYY-MM-dd')).getInfo())

# If STILL 0, widen by a few more days at a time (e.g. to '2024-10-08')
# rather than jumping straight back to 9+ days -- each step tells you
# exactly how many days post-peak the closest real pass actually is.

before_count = before_col.size().getInfo()
after_count = after_col.size().getInfo()
print(f"Before images found: {before_count}")
print(f"After images found: {after_count}")

if before_count == 0 or after_count == 0:
    raise RuntimeError("No images in one of the windows -- widen the date ranges above.")

before = before_col.median().clip(aoi)
after = after_col.median().clip(aoi)

# ----------------------------------------------------------------------
# 4. Change detection (same thresholds as the JS version -- tune if needed)
# ----------------------------------------------------------------------
before_filtered = before.focal_median(50, 'circle', 'meters')
after_filtered = after.focal_median(50, 'circle', 'meters')
diff = after_filtered.subtract(before_filtered)

DIFF_THRESHOLD = -3
ABSOLUTE_THRESHOLD = -17
flooded_raw = diff.lt(DIFF_THRESHOLD).And(after_filtered.lt(ABSOLUTE_THRESHOLD))

# ----------------------------------------------------------------------
# 5. Mask out permanent water
# ----------------------------------------------------------------------
permanent_water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').gt(50)
new_flood = flooded_raw.updateMask(permanent_water.Not())
flood_mask = new_flood.updateMask(new_flood.connectedPixelCount(8).gte(8))

# ----------------------------------------------------------------------
# 6. Vectorize and export DIRECTLY to a local file (geemap handles this,
#    no manual Google Drive download step needed)
# ----------------------------------------------------------------------
print("Vectorizing flood extent...")
flood_vector = flood_mask.selfMask().reduceToVectors(
    geometry=aoi, scale=10, geometryType='polygon', eightConnected=True, maxPixels=1e9
)

n_features = flood_vector.size().getInfo()
print(f"Flood polygons detected: {n_features}")

if n_features == 0:
    print("*** No flood polygons detected. Try loosening DIFF_THRESHOLD (-2) ")
    print("*** or ABSOLUTE_THRESHOLD (-15), or widen the date windows.")
else:
    OUT_PATH = "bharatpur_sar_flood_2024-09-28.geojson"
    geemap.ee_export_vector(flood_vector, filename=OUT_PATH)
    print(f"Saved flood extent directly to {OUT_PATH} -- no Drive download needed.")
    print("\nNext: run sar_intersect_grid.py pointing at this file,")
    print("then compute_sar_iou.py to compare against your model's predictions.")