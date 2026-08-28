# Data Pipeline

Everything that turns raw public geospatial and hydro-meteorological data into the
2.59M-row training set and the trained models the backend serves.

```
acquisition/   fetch raw data from BIPAD, Earth Engine, CDS, Sentinel-1
pipeline/      numbered stages 00 → 10, run in order
models/        training, threshold tuning, evaluation, figure generation
validation/    dataset checks, diagnostics, sanity tests
reference/     station directories, IDW weights, flood evidence records
samples/       small committed slices of datasets too large for version control
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GeoPandas, Rasterio and pysheds need GDAL available on the system. On macOS,
`brew install gdal` before installing the Python packages.

## Order of operations

| Stage   | Script                                         | Produces                                      |
| ------- | ---------------------------------------------- | --------------------------------------------- |
| —       | `acquisition/A_bipad_discover_stations.py`     | Station directory from the BIPAD API          |
| —       | `acquisition/B_bipad_bulk_download.py`         | Hourly rainfall, multi-station                |
| —       | `acquisition/D_bipad_river_bulk_download.py`   | Hourly river levels, Devghat and Rajaiya      |
| —       | `acquisition/acquire_chirps_via_earthengine.py`| CHIRPS daily rainfall, 1993–2020              |
| —       | `acquisition/acquire_glofas_via_cds.py`        | GloFAS-ERA5 river discharge, 1993–2020        |
| —       | `acquisition/sar_local_pipeline.py`            | Sentinel-1 flood extent, 2024-09-28           |
| 00      | `pipeline/00_assign_stations_to_grid.py`       | Grid → station assignment, IDW weights        |
| 01      | `pipeline/01_flood_evidence_and_static_finalize.py` | Static dataset with flood evidence       |
| 01b     | `pipeline/01b_recalibrate_risk_thresholds.py`  | Jenks natural-breaks risk class thresholds    |
| 02      | `pipeline/02_rainfall_event_selection.py`      | Rainfall event windows                        |
| 03      | `pipeline/03_build_ml_training_dataset.py`     | Observed-data training rows                   |
| 04      | `pipeline/04_train_models_v2.py`               | LogReg, Random Forest, XGBoost                |
| 05      | `pipeline/05_flow_accumulation_extraction.py`  | D8 flow accumulation from the DEM             |
| 06      | `pipeline/06_chirps_glofas_historical_backfill.py` | 1993–2020 reconstruction                  |
| 07      | `pipeline/07_desinventar_evidence_expansion.py`| Expanded historical flood evidence            |
| 08      | `pipeline/08_apply_historical_evidence_override.py` | Evidence overrides applied               |
| 09      | `pipeline/09_build_historical_grid_training_rows.py` | Per-grid historical rows                |
| 09a     | `pipeline/09a_stratify_sample_historical_events.py` | Stratified historical sample             |
| 10      | `pipeline/10_merge_and_finalize_training_data.py` | Final training set                         |

Then, under `models/`:

```bash
python models/tune_danger_threshold.py       # calibrate the Danger threshold
python models/save_tuned_model.py            # persist the deployed model
python models/evaluate_by_river.py           # stratified per-basin evaluation
python models/generate_report_graphs.py      # evaluation figures
```

## What is committed and what is not

**Committed** — every script, the reference tables, the BIPAD station directory, the
historical flood evidence records, and small samples.

**Not committed** — anything regenerable and large:

| Artefact                                 | Size    | How to get it                                 |
| ---------------------------------------- | ------- | --------------------------------------------- |
| `ml_training_dataset_v2.csv`             | ~1.5GB  | Run stages 00 → 10                            |
| `historical_grid_training_rows.csv`      | ~1.0GB  | Stage 09                                      |
| `ESA_WorldCover_10m_2021_v200_N27E084`   | ~1.3GB  | Download the N27E084 tile from ESA WorldCover |
| Copernicus DEM GLO-30, study area        | ~12MB   | Copernicus Data Space                         |
| `rainfall_hourly_multistation.csv`       | ~20MB   | `acquisition/B_bipad_bulk_download.py`        |
| `river_level_hourly_multistation.csv`    | ~28MB   | `acquisition/D_bipad_river_bulk_download.py`  |
| `model_random_forest.joblib`             | 384MB   | Stage 04                                      |

## Samples

`samples/ml_training_dataset_clean_sample.csv` is a 120,000-row slice of the final training
set carrying every feature column, enough to inspect the schema, reproduce the exploratory
figures, and sanity-check the feature engineering without the full 1.5GB file. The other
files in `samples/` are the static per-grid datasets at full size — they are small enough
to commit as they are.

## Data sources

| Source                | Access                                                                 |
| --------------------- | ---------------------------------------------------------------------- |
| BIPAD Portal (NDRRMA) | Public API — <https://bipadportal.gov.np>                              |
| DHM Nepal             | Station network, surfaced through BIPAD                                 |
| Copernicus DEM GLO-30 | Copernicus Data Space Ecosystem                                         |
| ESA WorldCover 2021   | <https://esa-worldcover.org> — tile N27E084                            |
| CHIRPS                | Google Earth Engine (`UCSB-CHG/CHIRPS/DAILY`) — needs an EE account     |
| GloFAS-ERA5           | Copernicus Climate Data Store — needs a CDS API key                     |
| Sentinel-1 GRD        | Google Earth Engine (`COPERNICUS/S1_GRD`)                               |

Earth Engine and CDS both require credentials. Configure them through their own CLI tools
(`earthengine authenticate`, `~/.cdsapirc`) — no credentials are read from this repository.
