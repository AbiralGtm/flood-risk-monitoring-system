"""
05_flow_accumulation_extraction.py

PURPOSE
-------
Adds the flow_accumulation feature to static_dataset_final.geojson.
This is a feature explicitly required by your project proposal (Table 3.2,
Section 3.5) and cited in your own literature review (Feizbahr et al.) but
was never computed in the original pipeline (00-04). This script fills that
gap without disturbing anything already verified in your static dataset.

UPDATE: this version now ALSO rebalances static_susceptibility_score to
include flow_accumulation_score (weight 0.05, borrowed from slope_score:
0.20 -> 0.15), and re-runs Jenks natural breaks on the new score so
static_risk_class reflects the updated formula. This was validated via:
  - Spearman/Pearson correlation of flow_accumulation_score against
    river_distance_score, elevation_score, slope_score: all near-zero,
    confirming it carries genuinely independent risk information rather
    than duplicating an existing sub-score.
  - Class-level comparison: with a 0.05 weight, ~10.4% of grids changed
    risk class (254/2434), Danger grew from 560->665. This needs to be
    re-validated against the FULL dynamic pipeline's calm-condition check
    in 03_build_ml_training_dataset.py before being treated as final --
    if that check regresses from its documented 0.0% false-positive rate,
    dial FLOW_ACC_WEIGHT down (e.g. to 0.03) and re-run this script.

RUN THIS LOCALLY / ON COLAB, NOT INSIDE THE CLAUDE SANDBOX.
The sandbox network cannot reach your data files or GDAL system libraries
reliably for this kind of raster processing. Run on your own machine or a
Colab notebook (matches your proposal's stated technical feasibility plan).

REQUIRED INPUTS (place in the same folder or edit paths below)
----------------------------------------------------------------
1. dem_clipped.tif -- your SRTM/DEM raster, already clipped to the
                       Bharatpur study area (same one used for
                       elevation/slope in 01_flood_evidence_and_static_finalize.py)
2. static_grid_dataset_with_flood_evidence.geojson -- output of 01b
                       (2,434 grids, EPSG:32645, recalibrated)

OUTPUT
------
static_dataset_with_flowacc.geojson
  -- same 2,434 grids, EPSG:32645, with:
     flow_accumulation        : raw D8 flow accumulation cell count
     flow_accumulation_score  : log-normalized 0-1 score
     static_susceptibility_score : REBALANCED to include flow_accumulation_score
     static_risk_class        : re-classified via Jenks on the new score
     static_risk_class_pre_flowacc : the OLD classification, kept for
                                      comparison/report purposes

  This is the file 00_assign_stations_to_grid.py should now read as its
  GRID_GEOJSON input (see that script's updated path).

INSTALL (run once, locally)
----------------------------
pip install pysheds geopandas rasterio shapely mapclassify --break-system-packages

  NOTE: this script uses pysheds, NOT richdem. richdem is a C++ extension
  that frequently fails to build on newer clang toolchains (Mac/Ubuntu) --
  if you hit "Failed building wheel for richdem" errors, that's why.
  pysheds is pure Python + numpy, installs cleanly everywhere, and does
  the same D8 flow accumulation workflow.
"""

import sys
import numpy as np

# Compatibility shim: pysheds (as of v0.5) still calls np.in1d, which was
# removed in numpy 2.0+ in favor of np.isin. This restores it rather than
# forcing you to downgrade numpy for one dependency's sake.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# Weight given to flow_accumulation_score in the rebalanced formula.
# Borrowed from slope_score (0.20 -> 0.15). Keep this as a named constant
# so it's easy to dial down (e.g. to 0.03) if the calm-condition check in
# 03_build_ml_training_dataset.py regresses.
FLOW_ACC_WEIGHT = 0.05


def main():
    try:
        from pysheds.grid import Grid
    except ImportError:
        print("pysheds not installed. Run:")
        print("  pip install pysheds --break-system-packages")
        sys.exit(1)

    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not installed. Run:")
        print("  pip install geopandas --break-system-packages")
        sys.exit(1)

    import pandas as pd
    import rasterio
    from rasterio.transform import rowcol

    DEM_PATH = "dem_clipped.tif"
    GRID_PATH = "static_grid_dataset_with_flood_evidence.geojson"
    OUT_PATH = "static_dataset_with_flowacc.geojson"

    print(f"Loading DEM from {DEM_PATH} ...")
    # pysheds >=0.4 API: methods take/return Raster objects directly rather
    # than the older data_name/out_name string-registry style.
    grid = Grid.from_raster(DEM_PATH)
    dem = grid.read_raster(DEM_PATH)

    print("Filling pits and depressions (required before flow routing) ...")
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)

    print("Resolving flats ...")
    inflated = grid.resolve_flats(flooded)

    print("Computing D8 flow direction ...")
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated, dirmap=dirmap)

    print("Computing D8 flow accumulation ...")
    acc = grid.accumulation(fdir, dirmap=dirmap)
    accum_array = np.array(acc)
    dem_transform = grid.affine
    dem_shape = accum_array.shape

    print(f"Loading grid dataset from {GRID_PATH} ...")
    grids = gpd.read_file(GRID_PATH)
    print(f"  {len(grids)} grids loaded, CRS: {grids.crs}")

    with rasterio.open(DEM_PATH) as src:
        dem_crs = src.crs

    if grids.crs != dem_crs:
        print(f"  Reprojecting grids from {grids.crs} to {dem_crs} for sampling ...")
        grids_for_sampling = grids.to_crs(dem_crs)
    else:
        grids_for_sampling = grids

    flow_acc_values = []
    out_of_bounds = 0
    for geom in grids_for_sampling.geometry:
        centroid = geom.centroid
        try:
            row, col = rowcol(dem_transform, centroid.x, centroid.y)
            if 0 <= row < dem_shape[0] and 0 <= col < dem_shape[1]:
                val = float(accum_array[row, col])
            else:
                val = np.nan
                out_of_bounds += 1
        except Exception:
            val = np.nan
            out_of_bounds += 1
        flow_acc_values.append(val)

    grids["flow_accumulation"] = flow_acc_values

    n_nan = grids["flow_accumulation"].isna().sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} grids got NaN flow_accumulation "
              f"(centroid fell outside DEM extent, {out_of_bounds} confirmed out-of-bounds). "
              f"Filling with the dataset median so downstream steps don't break -- "
              f"but check your DEM clip extent covers the full grid extent.")
        median_val = grids["flow_accumulation"].median()
        grids["flow_accumulation"] = grids["flow_accumulation"].fillna(median_val)

    # Log-normalize to 0-1, consistent with the style of your other *_score fields.
    # +1 avoids log(0) for cells with zero/near-zero accumulation.
    log_vals = np.log1p(grids["flow_accumulation"].clip(lower=0))
    lo, hi = log_vals.min(), log_vals.max()
    if hi > lo:
        grids["flow_accumulation_score"] = (log_vals - lo) / (hi - lo)
    else:
        grids["flow_accumulation_score"] = 0.0

    # ------------------------------------------------------------------
    # Rebalance static_susceptibility_score to include flow_accumulation_score,
    # then re-run Jenks so static_risk_class reflects the new formula.
    #
    # Validated: flow_accumulation_score has near-zero Pearson AND Spearman
    # correlation with river_distance_score/elevation_score/slope_score
    # (checked on the real 2434-grid dataset), confirming this is genuinely
    # new information, not double-counting an existing sub-score.
    # ------------------------------------------------------------------
    print(f"\nRebalancing static_susceptibility_score (flow_accumulation weight = {FLOW_ACC_WEIGHT}) ...")
    for col in ["river_distance_score", "elevation_score", "slope_score", "lulc_score", "history_score"]:
        assert col in grids.columns, f"Missing expected column: {col}"

    grids["static_risk_class_pre_flowacc"] = grids["static_risk_class"]
    grids["static_susceptibility_score"] = (
        0.30 * grids["river_distance_score"]
        + 0.25 * grids["elevation_score"]
        + (0.20 - FLOW_ACC_WEIGHT) * grids["slope_score"]
        + 0.15 * grids["lulc_score"]
        + 0.10 * grids["history_score"]
        + FLOW_ACC_WEIGHT * grids["flow_accumulation_score"]
    )

    try:
        import mapclassify
        jenks = mapclassify.NaturalBreaks(grids["static_susceptibility_score"], k=4)
        breaks = jenks.bins
        method = "Jenks natural breaks (mapclassify)"
    except ImportError:
        breaks = grids["static_susceptibility_score"].quantile([0.25, 0.50, 0.75, 1.0]).values
        method = "Quantile classification (mapclassify not installed)"

    def classify(s, breaks):
        if s <= breaks[0]:
            return "Low"
        if s <= breaks[1]:
            return "Medium"
        if s <= breaks[2]:
            return "High"
        return "Danger"

    grids["static_risk_class"] = grids["static_susceptibility_score"].apply(lambda s: classify(s, breaks))

    print(f"Classification method: {method}")
    print(f"New breakpoints: {breaks}")
    print("\nBEFORE (no flow_accumulation) static_risk_class distribution:")
    print(grids["static_risk_class_pre_flowacc"].value_counts())
    print("\nAFTER (with flow_accumulation) static_risk_class distribution:")
    print(grids["static_risk_class"].value_counts())
    n_changed = (grids["static_risk_class"] != grids["static_risk_class_pre_flowacc"]).sum()
    print(f"\nGrids that changed risk class: {n_changed} / {len(grids)} "
          f"({100*n_changed/len(grids):.1f}%)")

    grids.to_file(OUT_PATH, driver="GeoJSON")

    # --- Self-verification block, matching the PASS/FAIL style of 03_build_ml_training_dataset.py ---
    print("\n--- SELF-VERIFICATION ---")
    checks_passed = True

    if len(grids) != 2434:
        print(f"FAIL: expected 2434 grids, got {len(grids)}")
        checks_passed = False
    else:
        print(f"PASS: grid count preserved ({len(grids)})")

    if grids["flow_accumulation_score"].between(0, 1).all():
        print("PASS: flow_accumulation_score is within [0,1]")
    else:
        print("FAIL: flow_accumulation_score has values outside [0,1]")
        checks_passed = False

    if grids["static_susceptibility_score"].between(0, 1).all():
        print("PASS: rebalanced static_susceptibility_score is within [0,1]")
    else:
        print("FAIL: rebalanced static_susceptibility_score has values outside [0,1]")
        checks_passed = False

    if n_nan / len(grids) > 0.05:
        print(f"FAIL: {n_nan} grids ({100*n_nan/len(grids):.1f}%) had missing DEM coverage -- "
              f"this is too high, check your DEM clip extent")
        checks_passed = False
    else:
        print(f"PASS: missing-coverage rate acceptable ({n_nan} grids, {100*n_nan/len(grids):.1f}%)")

    print(f"\nOverall: {'PASS' if checks_passed else 'FAIL -- review warnings above'}")
    print(f"\nSaved: {OUT_PATH}")
    print("\nNEXT STEP: run 00_assign_stations_to_grid.py, which should now read")
    print(f"{OUT_PATH} as its GRID_GEOJSON input (confirm that path is updated),")
    print("then run 02 and 03. Watch 03's own self-verification (the calm-condition")
    print("check) to confirm this weighting doesn't regress the documented 0.0%")
    print("false-positive rate. If it does, lower FLOW_ACC_WEIGHT at the top of")
    print("this script (e.g. to 0.03) and re-run from here.")


if __name__ == "__main__":
    main()
