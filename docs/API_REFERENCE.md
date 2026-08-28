# Bharatpur Flood Risk — Frontend Context

This replaces the earlier draft frontend context doc. The API surface has since been built out (model integration, live prediction, model comparison, batch risk map, affected-area dissolve) and several field names/shapes in the original draft don't match what's actually implemented. This doc reflects the real backend as of this write-up, verified against a running instance.

## What this frontend needs to display

A grid-wise flood risk map for Bharatpur Metropolitan City, Chitwan, Nepal — 250m x 250m grid cells (2,575 of them), each colored by predicted flood risk class, with a control for entering current/forecast rainfall and river conditions.

This is a decision-support prototype, not an official DHM warning system. If there's a disclaimer/about section, it should say so.

## Base URL & CORS

Local dev: `http://localhost:8000`. Everything is under `/api` except `GET /` and `GET /health`. CORS is currently allowed from `localhost:3000`, `127.0.0.1:3000`, `localhost:5173`, `127.0.0.1:5173` (`app/config.py` — add your dev port there if different).

## Coordinate system (fixed, but know this)

The source GeoJSON is in EPSG:32645 (UTM Zone 45N, meters), not WGS84. Every endpoint that returns GeoJSON now reprojects to EPSG:4326 (lon/lat) automatically at load time, so Leaflet will render it correctly with no frontend-side reprojection needed. Verified: grid centroids land at ~84.16°E, 27.57°N as expected for Bharatpur.

## The core design point: risk is not static

`predicted_risk_class` depends on live rainfall/river conditions, not just terrain. There is no single "current risk map" sitting on the server — every map render is a request that says "score all grids under these conditions." Call `GET /api/conditions/live` first to get real current numbers, then feed those straight into `/api/risk-map`.

Two distinct risk concepts exist, don't conflate them in the UI:
- **Static susceptibility** (`static_susceptibility_class`) — terrain/land-cover only (elevation, slope, river distance, land cover). Doesn't change per request. Good for a baseline/"which areas are inherently vulnerable" view.
- **Predicted risk** (`predicted_risk_class`) — the actual XGBoost/RF/LogReg model output, requires rainfall + water-level inputs. This is the real flood risk signal and what the color-coded map should primarily show.

## Suggested data flow

1. On load: call `GET /api/conditions/live` to get real current rainfall/river numbers, pre-fill the conditions form with them, then call `GET /api/risk-map` with those values to get the full colored grid. If `/api/conditions/live` returns a 502 (BIPAD unreachable), fall back to a manual-entry form — don't silently treat that as "conditions are calm."
2. Grid click: read the popup fields directly off the already-loaded feature's `properties` — no extra request needed, `/api/risk-map` already merges static properties with the prediction.
3. "What if" / live re-check on one grid: `POST /api/predict` (best model) or `POST /api/predict/compare` (all 3, for a model-comparison panel).
4. Rainfall/river input form changes: re-call `GET /api/risk-map` with the new values to refresh the whole map.
5. Model metadata / accuracy table (for an about/methodology panel): `GET /api/models`.
6. "Affected area" boundary overlay: `GET /api/risk-map/affected-area` with the same conditions.

## Endpoints

### `GET /api/conditions/live` — real current rainfall + river level

No required params (`river_station_id` optional, `devghat` default or `rajaiya`). Fetches from BIPAD's public API, cached 5 minutes server-side:

```json
{
  "rainfall_1h_mm": 0.0, "rainfall_3h_mm": 0.0, "rainfall_6h_mm": 0.0,
  "rainfall_12h_mm": 1.2, "rainfall_24h_mm": 12.4,
  "water_level_m": 4.72,
  "warning_level_m": 7.3, "danger_level_m": 9.0,
  "river_station_id": "devghat",
  "rain_stations_used": ["Rampur_AWS", "Narayani Field Office", "Meghauli"],
  "source": "bipad",
  "fetched_at": "2026-07-27T19:32:10.123456+05:45"
}
```

Rainfall is averaged across whichever of the 3 active BIPAD rain gauges inside Bharatpur municipality have a fresh reading (`rain_stations_used` tells you which). `water_level_m`/`warning_level_m`/`danger_level_m` come straight from the chosen river gauge. Feed `warning_level_m`/`danger_level_m` straight through as the optional params on `/api/risk-map` for a more accurate `river_level_score` than the 0.3 fallback.

Returns `502` if BIPAD is unreachable — treat that as "live data unavailable," not "conditions are calm," and fall back to manual entry.

### `GET /api/conditions/forecast?horizon=24h|48h|72h` — Forecast Mode's rainfall source

Required `horizon` param, one of `24h`/`48h`/`72h`. Fetches from Open-Meteo's free public forecast API (real NWP-model-based forward-looking forecast, not an estimate of rain that already fell), cached 30 minutes server-side per horizon:

```json
{
  "horizon": "48h",
  "rainfall_24h_mm": 38.4,
  "window_start": "2026-08-04T20:00",
  "window_end": "2026-08-06T19:00",
  "source": "open-meteo",
  "fetched_at": "2026-08-04T19:55:02.123456+05:45"
}
```

`rainfall_24h_mm` is named to match the model's actual feature — it's the **cumulative rainfall forecast for the entire horizon window**, not literally "next 24 hours" when horizon is `48h`/`72h`. Pass it straight through as the `rainfall_24h_mm` value in a `/api/predict` or `/api/risk-map` conditions payload representing "best estimate for this horizon." There is no per-horizon-aware model (single `rainfall_24h_mm` input regardless of horizon — see `app/ml/features.py`), so two different horizons submitted with the same rainfall number will still produce identical predictions; that's expected, not a bug.

No river-level forecast source exists — Open-Meteo doesn't provide one. Keep using `GET /api/conditions/live`'s `water_level_m` (or a manually entered value) for the river side of a forecast scenario.

Returns `502` if Open-Meteo is unreachable — same rule as `/conditions/live`: never silently substitute a guessed rainfall value.

### `GET /api/risk-map` — the main map data source

Query params:

| param | type | notes |
|---|---|---|
| `model` | string, optional | `xgboost_tuned` (default, best model, 2026-08-08+) / `xgboost` / `random_forest` / `logreg` |
| `rainfall_24h_mm` | float, required | mm accumulated in the last 24h (the only rainfall window the model uses — the old `rainfall_1h_mm`/`3h_mm`/`6h_mm`/`12h_mm` params are gone; harmless to still send them, FastAPI just ignores undeclared params) |
| `water_level_m` | float, optional (deprecated) | applied uniformly to every grid, regardless of which river it's near. Kept for backward compatibility only — see `water_level_rv01_m`/`water_level_rv02_m` below. |
| `water_level_rv01_m` | float, optional | **Preferred.** Narayani/Devghat gauge reading — applied only to grids near the Narayani. |
| `water_level_rv02_m` | float, optional | **Preferred.** East Rapti/Rajaiya gauge reading — applied only to grids near the Rapti. |
| `river_station_id`, `warning_level_m`, `danger_level_m` | optional | see caveats below — `river_station_id` no longer does anything at this endpoint |

**At least one of `water_level_m`, `water_level_rv01_m`, `water_level_rv02_m` is required** — 422 if all three are omitted.

**Fixed 2026-08: grids now resolve their own river automatically.** Every grid has its own real, pre-assigned `river_station_id` ("RV01" for Narayani-side grids, "RV02" for Rapti-side — 1,298 and 1,136 grids respectively). Previously, one global `water_level_m`/`river_station_id` was applied identically to all 2,434 grids, which made it architecturally impossible to represent "Narayani rising, Rapti flat" — a rising Narayani reading bled into every Rapti-side grid's score too, and (separately) `river_station_id` values in the RV01/RV02 format never matched the lookup table's place-name keys at all, silently falling through to a flat fallback score. Both are fixed: pass `water_level_rv01_m`/`water_level_rv02_m` for genuine per-river conditions, and each grid now automatically uses its own correct station's thresholds — you don't need to pass `river_station_id` at all for this to work correctly (it's still accepted so old calls don't break, but at this endpoint it no longer forces every grid onto one station — see `POST /api/predict` below for where an explicit station override still does something).

`warning_level_m`/`danger_level_m`, if passed, still apply **globally** to every grid regardless of river (unchanged, not part of this fix) — only pass these if you deliberately want to override every grid's threshold at once; leave them out to let each grid use its own correct built-in thresholds (7.3/9.0 for Narayani, 3.3/3.7 for Rapti).

Example (new, preferred):
```
GET /api/risk-map?model=xgboost&rainfall_24h_mm=25&water_level_rv01_m=8.8&water_level_rv02_m=2.0
```

Returns a GeoJSON FeatureCollection. Each feature's `properties` contains **all** static grid fields (below) plus:

```
predicted_risk_class   "Low" | "Medium" | "High" | "Danger"
risk_probability        0-1, confidence in predicted_risk_class
risk_probabilities      {"Low":.., "Medium":.., "High":.., "Danger":..}
model_id                which model produced this
water_level_m            this grid's own RESOLVED water level (can now differ grid-to-grid)
river_station_id         this grid's own resolved station ("RV01"/"RV02")
river_level_score, river_level_score_used_fallback  (see below)
```

Also a top-level `skipped_grids` array: `[{grid_id, reason}, ...]` — currently always empty against the canonical 2,434-grid dataset (`data/static_grid_dataset_with_stations.geojson`), since that file's population filter is already applied upstream; kept in the response in case a future dataset swap reintroduces genuine per-grid gaps, so a silently-dropped grid would be visible here rather than just missing.

**BREAKING CHANGE:** the old top-level `river_level_score_used_fallback` boolean is gone, replaced with `grids_using_river_fallback` (an integer count). The old boolean only ever made sense because every grid shared one condition — now that grids can resolve to different stations/readings, "was the fallback used" isn't a single yes/no for the whole map. Read the per-feature `properties.river_level_score_used_fallback` if you need to know about one specific grid, or the new top-level count for an at-a-glance map-wide check.

### Static grid properties (present on every feature, from the source GeoJSON)

```
grid_id, elevation_mean, elevation_min, elevation_max, slope_mean,
distance_to_narayani_m, distance_to_rapti_m, nearest_river_distance,
nearest_major_river ("Narayani" | "Rapti"), within_2km_narayani, within_2km_rapti,
lulc_dominant, lulc_encoded, lulc_score,
elevation_score, slope_score, river_distance_score,
area_km2, centroid_lat, centroid_lon
```

Note vs. the earlier draft: the field is `elevation_mean` not `elev_mean`. There is no per-grid `rainfall_24h_mm` stored in the static dataset — it's a request input, not a grid property (same value applies to every grid in one request; if you want it in the popup, just read it from the form state you already have, not from the feature). `historical_flood_evidence` **is** present in the current canonical dataset (118 of 2,434 grids carry `1`; this doc previously said it wasn't in the dataset at all — that was true of an earlier data file, not the current one) — its trained feature importance is still small (~0.003), so it rarely swings a prediction on its own, but it's real per-grid data now, not always `0`.

### Grid click popup — recommended fields (all come from the already-loaded `/api/risk-map` feature, no extra call needed)

```
grid_id, predicted_risk_class, risk_probability,
elevation_mean, slope_mean, distance_to_narayani_m, distance_to_rapti_m,
nearest_major_river, lulc_dominant
```

`risk_probability` (0-1) is the model's confidence — good for opacity or a secondary badge, as originally planned.

### `POST /api/predict` — single-grid prediction, best model only

```json
// Request
{
  "grid_id": "GRID_617",
  "conditions": {
    "rainfall_24h_mm": 145.0, "water_level_m": 8.1,
    "river_station_id": "devghat", "warning_level_m": 7.3, "danger_level_m": 9.0
  }
}
// Response
{
  "grid_id": "GRID_617",
  "model_id": "xgboost",
  "predicted_risk_class": "High",
  "risk_probability": 0.87,
  "probabilities": {"Low": 0.02, "Medium": 0.11, "High": 0.87, "Danger": 0.00},
  "diagnostics": {
    "river_level_score": 0.62,
    "river_level_score_used_fallback": false,
    "river_station_id": "devghat"
  },
  "contributing_factors": [
    {"factor": "rainfall", "label": "Very heavy rainfall (145mm in 24h)", "severity": "high"},
    {"factor": "river_level", "label": "River level approaching the warning threshold", "severity": "moderate"},
    {"factor": "historical_flood_evidence", "label": "This location has recorded historical flood evidence", "severity": "high"}
  ]
}
```

Differences from the earlier draft: the dynamic inputs are nested under `"conditions"`, not flat at the top level (and the old `rainfall_1h_mm`/`3h_mm`/`6h_mm`/`12h_mm` fields are gone — the model only takes `rainfall_24h_mm` now, see the model-retrain note below); `water_level_m` is required (not nullable); the response field is `risk_probability`, not `confidence`; `nearest_major_river`/distance aren't echoed back (you already have them from the grid's static properties). 404 if `grid_id` doesn't exist, 422 if that grid is outside the trained population (a sliver, area < 0.005 km²) or has missing static data (one of the 8 genuine gaps).

**Fixed 2026-08: you no longer have to pass `river_station_id` for this to work.** It used to be required (in practice) — omitting it, or passing a grid's own `"RV01"`/`"RV02"` value instead of a recognized place name, silently fell through to a flat `0.3` fallback score, with `water_level_m` barely moving the prediction. Both are fixed: if you omit `river_station_id` entirely, the backend now falls back to *this specific grid's own* correct, pre-assigned station automatically, and `"RV01"`/`"RV02"` are now recognized directly (aliased to the same thresholds as `"devghat"`/`"rajaiya"`). You can still pass an explicit `river_station_id` to override which station's thresholds this one grid's prediction uses (a legitimate single-grid what-if query) — that still takes priority over the grid's own value. `diagnostics.river_level_score_used_fallback` tells you when the fallback was used (should now be rare — only when a grid genuinely has no resolvable station at all); `diagnostics.river_station_id` shows the station actually used, whether from your request or resolved automatically. **Build the conditions object once, in one place in your app state, and pass that same object to every call** (`/api/risk-map`, `/api/predict`, `/api/predict/compare`) — two calls that assemble their `conditions` independently are how you get the map and the detail panel disagreeing on the same grid at the same time even though both hit the same backend function.

**`contributing_factors`** (proposal §3.7's "rule-based explanation") is a plain-language list of the specific factors pushing this grid toward higher risk — deterministic threshold rules (heavy rainfall, river level, river proximity, low elevation, high flow accumulation, historical flood evidence), not a statistical/SHAP-style attribution. Ordered most-severe-first (`severity`: `"high"` > `"moderate"` > `"low"`); render as-is, e.g. a "Why this risk level" list under the probability breakdown — that's literally what it's for. Only factors that actually trigger appear, so a low-risk grid under calm conditions can return a short list (in practice every grid in this dataset is within 2km of a major river by definition, so you'll always see at least that one low-severity entry — a fully empty list is possible in principle but won't happen with this study area's grids). The river-level factor is suppressed entirely (not shown, not caveated) whenever `diagnostics.river_level_score_used_fallback` is `true`, since attributing risk to a generic 0.3 fallback would misrepresent an estimate as a real reading.

### `POST /api/predict/compare` — same request shape, runs all 4 models

```json
{
  "grid_id": "GRID_382",
  "best_model": "xgboost_tuned",
  "predictions": {
    "xgboost_tuned": {"model_id": "xgboost_tuned", "predicted_risk_class": "High", "risk_probability": 0.508, "probabilities": {...}},
    "xgboost":       {"model_id": "xgboost", "predicted_risk_class": "High", "risk_probability": 0.508, "probabilities": {...}},
    "random_forest": {"model_id": "random_forest", "predicted_risk_class": "High", "risk_probability": 0.583, "probabilities": {...}},
    "logreg":        {"model_id": "logreg", "predicted_risk_class": "Medium", "risk_probability": 0.61, "probabilities": {...}}
  },
  "diagnostics": {"river_level_score": 0.3, "river_level_score_used_fallback": true, "river_station_id": null},
  "contributing_factors": [{"factor": "river_proximity", "label": "Within 2km of a major river", "severity": "low"}]
}
```

**Breaking change 2026-08-08: a 4th model, `xgboost_tuned`, now appears in `predictions`, and `best_model` is `"xgboost_tuned"` instead of `"xgboost"`.** If a model-comparison panel has 3 hardcoded columns/cards, it now needs a 4th — see `XGBOOST_TUNED_FRONTEND_PROMPT.md`. `xgboost_tuned` and `xgboost` will very often agree exactly (same underlying trained model, `xgboost_tuned` just applies a stricter Danger-classification gate on top — see the `GET /api/models` section above), so don't be surprised to see identical `predicted_risk_class`/`risk_probability` for both in most requests; they diverge specifically in borderline cases where plain `xgboost` would have called Danger at moderate confidence and `xgboost_tuned` calls something else instead.

Same `diagnostics` block as `/api/predict` — same warning applies. `contributing_factors` is also the same shape/meaning as `/api/predict`'s, but note it's a **single top-level array here, not one per model** — the factors describe the input feature values (which are identical across all four models), not any one model's output, so computing it four times would just be the same list repeated.

Use this for a model-comparison panel on grid click (side-by-side class + confidence per model), with the `xgboost_tuned` result visually marked as the authoritative one (`best_model`). If a model fails to load/predict (e.g. a library version mismatch), its entry is `{"model_id": ..., "error": "..."}` instead of failing the whole response — handle that gracefully in the UI (e.g. gray out that model's card). All four load and predict successfully as of 2026-08-08 (a pre-existing scikit-learn version-skew issue that broke `logreg`'s live predictions specifically has been fixed).

### `GET /api/models` — model metadata for an about/methodology panel

**Breaking change 2026-08-08: a 4th model, `xgboost_tuned`, is now the default/best model** (`best_model` changed from `"xgboost"` to `"xgboost_tuned"`). If the frontend hardcodes `"xgboost"` as a default anywhere instead of reading `best_model` from this response, update it — see `XGBOOST_TUNED_FRONTEND_PROMPT.md` for the full picture.

```json
{
  "best_model": "xgboost_tuned",
  "models": [
    {
      "model_id": "xgboost_tuned", "display_name": "XGBoost (Fine-Tuned)", "algorithm": "XGBClassifier",
      "is_best": true, "classes": ["Danger","High","Low","Medium"],
      "metrics": {
        "test_accuracy": 0.969, "macro_f1": 0.8968,
        "danger_recall": 0.9401, "danger_precision": 0.6471, "roc_auc": 0.9975,
        "confusion_matrix": [[62280,545,0,0],[197,12405,1118,0],[0,160,9075,625],[0,0,73,1146]]
      }
    },
    {
      "model_id": "xgboost", "display_name": "XGBoost", "algorithm": "XGBClassifier",
      "is_best": false, "classes": ["Danger","High","Low","Medium"],
      "metrics": {
        "test_accuracy": 0.9665, "macro_f1": 0.8816,
        "danger_recall": 0.959, "danger_precision": 0.5739, "roc_auc": 0.9975,
        "confusion_matrix": [[62280,545,0,0],[197,12403,1118,2],[0,160,8834,866],[0,0,50,1169]]
      }
    },
    {"model_id": "random_forest", "display_name": "Random Forest", "algorithm": "RandomForestClassifier", "is_best": false, "metrics": {"test_accuracy": 0.9224, "macro_f1": 0.7775, ...}},
    {"model_id": "logreg", "display_name": "Logistic Regression", "algorithm": "LogisticRegression", "is_best": false, "metrics": {"test_accuracy": 0.7733, "macro_f1": 0.5021, ...}}
  ]
}
```

`xgboost_tuned` is the same underlying trained XGBoost model as plain `xgboost` (identical weights) with one difference: predictions only classify a grid as `Danger` when the model's own `P(Danger)` clears a calibrated threshold (0.5936); otherwise the prediction falls back to the best of the other three classes. This trades a little Danger recall for meaningfully higher Danger precision (fewer false Danger alarms) — see the metrics above. All four models now have real validated metrics (`confusion_matrix` is a 4×4 array in `["Low","Medium","High","Danger"]` row/column order — true class on rows, predicted on columns). These come from a reproducible eval script (`app/ml/evaluate.py`), not a live computation — they only change when someone re-runs it after a model retrain.

### `GET /api/models/evaluation-info` — when were those metrics computed?

```json
{
  "available": true,
  "computed_at": "2026-08-07T09:47:14.633264+00:00",
  "dataset_path": "dataset/ml_training_dataset_v2.csv",
  "dataset_row_count": 2589288,
  "train_row_count": 2328850, "val_row_count": 172814, "test_row_count": 87624,
  "test_rows_with_nan_river_level_score": 22070, "river_level_score_fill_value": 0.3,
  "models_evaluated": ["xgboost", "random_forest", "logreg"]
}
```

Not required for anything in the current UI — useful if you want a small "last evaluated on ..." caption near the metrics in an about/methodology panel, so stale numbers after a future retrain are visible rather than silently assumed current. `available: false` (with a `message`) if the eval script has never been run in this environment.

### `GET /api/models/gis-output-test` — predicted zones vs. historical flood extent

Proposal §3.13.2: "Predicted High and Danger zones will be compared with historical flood areas where available. Spatial overlap may be evaluated using intersection area or Intersection over Union." Not a live computation — reads a results file written by `python -m app.ml.gis_output_test`, which scores the live XGBoost model against every historical flood event row in the training dataset (1993–2020, 700 events, 1,703,800 grid×event rows) and computes IoU/precision/recall of predicted High+Danger against recorded flood_label.

```json
{
  "available": true,
  "computed_at": "2026-08-07T11:03:44.025943+00:00",
  "results": {
    "combined_all_historical_rows": {"row_count": 1703800, "event_count": 700, "intersection_over_union": 0.8166, "precision": 0.9743, "recall": 0.8345},
    "historical_evidence_override_only": {"row_count": 73020, "event_count": 30, "intersection_over_union": 0.2902, "precision": 0.9904, "recall": 0.291},
    "reconstructed_only": {"row_count": 1630780, "event_count": 670, "intersection_over_union": 0.9545, "precision": 0.973, "recall": 0.9804}
  }
}
```

Three numbers, not one — read `methodology_note` in the actual response before displaying any of these. Most flood_label rows in this dataset (`reconstructed`, ~96%) came from the same formula/features the model trained on, so `reconstructed_only` mostly measures whether the model reproduces its own label rule, not real-world accuracy. `historical_evidence_override_only` (~4% of rows) is the methodologically clean subset — flood_label there came from actual outside evidence — and it tells a different story: precision is still high (0.99, few false alarms) but recall drops to 0.29, meaning the model misses about 71% of independently-verified historical flood grids. If you surface this in an about/methodology panel, show the override-only numbers as the honest figure, not `combined_all_historical_rows`, and consider a one-line caveat next to it. `available: false` (with a `message`) if the test has never been run in this environment.

### `GET /api/risk-map/affected-area` — dissolved High+Danger boundary

Same required query params as `/api/risk-map`. Returns one GeoJSON `Feature` (not a FeatureCollection):

```json
{"type": "Feature", "properties": {"model_id": "xgboost", "grid_count": 2107}, "geometry": {"type": "MultiPolygon", "coordinates": [...]}}
```

`geometry` is `null` if nothing is currently High/Danger. Good for a single highlighted "flood risk zone" outline layer instead of rendering 2000+ individual grid outlines.

### History endpoints — new, backed by a database (not required for the map to work)

These are new and optional for the frontend — nothing above depends on them. They read back rows that get written automatically, server-side, whenever the corresponding live endpoint succeeds:
`GET /api/conditions/live` → logs a rainfall + a river-level reading. `GET /api/conditions/forecast` → logs a forecast fetch. `POST /api/predict` → logs the prediction. `GET /api/risk-map` → logs every High/Danger grid in the response, debounced (see below) — this is also this project's alert history. (`POST /api/predict/compare` does NOT log — redundant with `/predict`.)

All five endpoints below are read-only, return most-recent-first, and take an optional `limit` (default 50, max 500):

- `GET /api/history/rainfall?limit=50`
- `GET /api/history/river-level?station_id=devghat&limit=50` (`station_id` optional)
- `GET /api/history/forecasts?horizon=24h&limit=50` (`horizon` optional, one of `24h`/`48h`/`72h`)
- `GET /api/history/predictions?grid_id=GRID_1&limit=50` (`grid_id` optional)
- `GET /api/history/risk-results?grid_id=GRID_1&run_id=<uuid>&limit=50` (`grid_id` and `run_id` both optional) — every High/Danger grid from a `/api/risk-map` call, i.e. the map-wide alert picture, not just grids someone clicked into. `run_id` groups every row written by the same risk-map call, so you can pull "everything that was alert-worthy in this one map refresh."

**Debounce note**: `/api/risk-map` gets hit on every map refresh, so writes are debounced server-side — a repeat call with essentially the same conditions (rainfall/water_level rounded to 1 decimal, same model) within 60 seconds of the last logged one won't write again. This means `/history/risk-results` reflects "alert-worthy grids over time," not a complete log of every single risk-map request — don't build anything that assumes 1:1 coverage.

One consequence worth knowing: a single severe-conditions risk-map call can log a very large number of rows at once (one per High/Danger grid — potentially 1000+ out of 2,434 grids in an extreme scenario), all sharing one `run_id`. That's expected, not a bug — group by `run_id` if you want "this map refresh's alerts" as one unit rather than a flat list.

Example response shape (`/api/history/risk-results`):

```json
{
  "count": 1,
  "records": [
    {
      "id": 1, "grid_id": "GRID_1", "model_id": "xgboost",
      "predicted_risk_class": "Danger", "rainfall_24h_mm": 250.0, "water_level_m": 11.0,
      "run_id": "f61c9640-3ee5-4611-b536-20bf893df26f",
      "created_at": "2026-08-06T22:54:02.083928"
    }
  ]
}
```

Example response shape (`/api/history/predictions`):

```json
{
  "count": 1,
  "records": [
    {
      "id": 1, "grid_id": "GRID_1", "model_id": "xgboost",
      "predicted_risk_class": "Medium", "risk_probability": 0.635974,
      "river_level_score": 0.190411, "river_level_score_used_fallback": false,
      "rainfall_24h_mm": 45.0, "water_level_m": 6.1,
      "created_at": "2026-08-04T17:03:22.365797"
    }
  ]
}
```

Good for a small "recent activity" panel or a rainfall/river-level trend chart. Logging is best-effort server-side (a DB hiccup never fails the live/predict response), so don't assume every live/predict call produced a row — treat these as "recent history," not a guaranteed complete audit trail.

### `GET /api/summary` — static susceptibility aggregate (not live risk)

```json
{
  "total_grids": 2575,
  "static_risk_class_counts": {"Low": 1589, "Medium": 445, "High": 391, "Danger": 150},
  "lulc_dominant_counts": {"Cropland": 1204, "Tree cover": 1144, ...},
  "average_elevation": 178.58, "average_slope": 2.285, "average_distance_to_river": 978.76,
  "min_elevation": 142.45, "max_elevation": 304.59, "min_slope": 0.046, "max_slope": 30.77
}
```

This is the static susceptibility distribution (terrain only), not live predicted risk under current conditions — there's no live equivalent endpoint. If you want a "current risk breakdown" stat card, compute it client-side by counting `predicted_risk_class` across the `/api/risk-map` response you already fetched.

### `GET /api/grids`, `GET /api/grids/{grid_id}`, `GET /api/grids/filter`, `GET /api/risk/static`, `GET /api/filters/lulc`, `GET /api/filters/risk-classes`

Raw/static-only views (no live prediction). `risk_class` filtering on `/api/grids/filter` and `/api/risk/static` uses `static_susceptibility_class`, not `predicted_risk_class` — same static-vs-live distinction as above applies.

## Map layers (per original spec)

- **Grid colored by `predicted_risk_class`** — from `/api/risk-map`, ready now.
- **Affected-area boundary** — from `/api/risk-map/affected-area`, ready now.
- **Narayani corridor, Rapti corridor** — `GET /api/layers/rivers`, ready now (see below).
- **Administrative (ward) boundaries** — `GET /api/layers/wards`, ready now (see below).
- **Ward-wise risk ranking/reporting** — `GET /api/wards/risk-report`, ready now (see below).
- **Bharatpur river-influence boundary** (a separate city-wide background boundary, distinct from the two river polygons) — not served by this API; no such geometry exists in the backend data yet.
- **Flood incident points (historical evidence)** — not currently servable. The flood evidence CSV referenced in the project docs isn't in this backend's `data/` folder and isn't wired to an endpoint. Flag as a future addition if you need this overlay.

### `GET /api/layers/rivers` — river corridor reference layer

No query params. Returns a static GeoJSON FeatureCollection (WGS84), 2 features:

```json
{
  "type": "FeatureCollection",
  "features": [
    {"type": "Feature", "properties": {"river_name": "Narayani"}, "geometry": {"type": "MultiPolygon", "coordinates": [...]}},
    {"type": "Feature", "properties": {"river_name": "Rapti"}, "geometry": {"type": "MultiPolygon", "coordinates": [...]}}
  ]
}
```

This is a fixed reference layer (draw once, doesn't need refetching on condition changes) — good as a low-opacity blue fill/outline under the risk grid. Note: coordinates carry a third (altitude) value of `0.0` per point from the source export; harmless for Leaflet, which ignores it.

### `GET /api/layers/wards` — administrative ward boundaries

No query params. Returns a static GeoJSON FeatureCollection (WGS84), 29 features, one per Bharatpur Metropolitan City ward:

```json
{
  "type": "FeatureCollection",
  "features": [
    {"type": "Feature", "properties": {"ward_number": 21, "municipality": "Bharatpur", "district": "CHITAWAN", "area_sqkm": 10.0162016913}, "geometry": {"type": "MultiPolygon", "coordinates": [...]}}
  ]
}
```

Another fixed reference layer, same pattern as `/api/layers/rivers` — draw once, low-opacity outline/labels, no refetching on condition changes. Good for a ward-number label layer or a ward-level "which areas are affected" summary if the frontend wants to aggregate risk grids by ward.

Coordinate-system note: the source file this was built from carried no `crs` member, but its raw coordinates were clearly projected meters, not degrees. The reprojection was calibrated by matching this file's bounding box against the grid dataset's already-correctly-labeled EPSG:32645 coordinates (their shared edge lined up to within ~2 meters) and cross-checked against the ward areas summing to 432.95 km², matching Bharatpur Metropolitan City's known official area. Flag if any ward boundary looks visibly off on the map — the calibration is empirical, not from an authoritative source CRS definition.

### `GET /api/wards/risk-report` — ward-wise reporting + risk ranking

Same query params as `/api/risk-map`, including the new per-station ones — `rainfall_24h_mm` (required), `water_level_rv01_m`/`water_level_rv02_m` (preferred, per-river), `water_level_m` (legacy, applied uniformly), optional `model`/`river_station_id`/`warning_level_m`/`danger_level_m`, and the same "at least one water level param required" validation. Ward risk is just as condition-dependent as grid risk, there's no static "riskiest ward" number sitting on the server — and now that grids can resolve genuinely different per-river readings, so can wards (a ward near the Rapti will correctly reflect a Rapti-only surge instead of being dragged up by an unrelated Narayani reading).

```json
{
  "model_id": "xgboost",
  "conditions": {"rainfall_24h_mm": 250.0, "water_level_m": null, "water_level_by_station": {"rv01": 11.0, "rv02": 4.0}, "river_station_id": null, "warning_level_m": null, "danger_level_m": null},
  "total_grids_scored": 2434,
  "skipped_grid_count": 0,
  "unassigned_grid_count": 0,
  "wards": [
    {
      "rank": 1, "ward_number": 28, "municipality": "Bharatpur", "district": "CHITAWAN", "area_sqkm": 8.91,
      "grid_count": 212, "risk_class_counts": {"Low": 0, "Medium": 0, "High": 1, "Danger": 211},
      "high_danger_count": 212, "high_danger_pct": 100.0, "risk_score": 2.9953, "dominant_risk_class": "Danger"
    }
  ],
  "wards_with_no_grid_coverage": [7, 9, 12, 20, 21, 29]
}
```

`wards` is pre-sorted, riskiest first (`rank: 1` = highest risk) — render top-to-bottom as-is for a leaderboard/table, no client-side re-sorting needed. `risk_score` is a 0–3 severity average (Low=0 ... Danger=3) across each ward's grids, used for ranking; `high_danger_pct` is the more human-readable "% of ward at High/Danger" number, good for a leaderboard column or a choropleth fill scale on the ward polygons from `/api/layers/wards` (join on `ward_number`).

`wards_with_no_grid_coverage` lists wards with **zero** grids in the model's study-area population — the flood-relevant grid coverage is filtered to the area near the rivers (see the "Known gaps" section), not the full municipal footprint, so several wards farther out legitimately have nothing to score. Show these as "no data" / grayed out, not as "0% risk" — there's a real difference between "verified low risk" and "not covered by this dataset."

Good uses: a ward leaderboard/table (sort already done server-side), or a ward-level choropleth layer (color `/api/layers/wards` polygons by each ward's `risk_score`/`high_danger_pct` from this response, joined on `ward_number`) as an alternative zoomed-out view to the 2,434-grid map.

## Risk color scheme (exact, unchanged from spec)

```
Low     -> green   (#2ecc71)
Medium  -> yellow  (#f1c40f)
High    -> orange  (#e67e22)
Danger  -> red     (#e74c3c)
```

## Tech stack

React + Leaflet.js (or Mapbox GL JS) per the original proposal. Coordinates are now plain WGS84 GeoJSON, so `L.geoJSON()` works directly with no CRS plugin needed.

## Known gaps / caveats to carry into the UI

- 141/2,575 grids (~5.5%) are excluded from every prediction — 133 slivers (area < 0.005 km², matches the trained model's population exactly) plus 8 genuine data gaps — surfaced via `skipped_grids` on `/api/risk-map` with a distinct reason for each.
- `historical_flood_evidence` is always `0` server-side (not in the current dataset); documented as low-impact given its ~0.003 trained feature importance.
- Random Forest and Logistic Regression have no validated accuracy/F1 numbers yet (`null` in `/api/models`) — only XGBoost's are real.
- `GET /api/conditions/live` depends on BIPAD's public API staying available and stable — it's undocumented (found by inspecting network requests, not a published contract), so it could change without notice. Always have a manual-entry fallback in the UI.
