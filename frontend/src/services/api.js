const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const buildUrl = (path, params) => {
  const url = new URL(path, API_BASE_URL);

  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== "") {
        url.searchParams.append(key, value);
      }
    });
  }

  return url.toString();
};

const parseJsonOrThrow = async (response, path) => {
  const rawText = await response.text();

  if (!response.ok) {
    // Try to surface a useful API error message (FastAPI 422s include `detail`).
    let detail = response.statusText;
    try {
      const parsed = JSON.parse(rawText);
      detail = parsed?.detail ? JSON.stringify(parsed.detail) : detail;
    } catch (parseError) {
      // rawText wasn't JSON, fall back to statusText
    }
    const error = new Error(`API request failed (${response.status}) for ${path}: ${detail}`);
    error.status = response.status;
    throw error;
  }

  try {
    return JSON.parse(rawText);
  } catch (error) {
    throw new Error(`Invalid JSON response from ${path}`);
  }
};

const requestJson = async (path, params) => {
  const response = await fetch(buildUrl(path, params));
  return parseJsonOrThrow(response, path);
};

const postJson = async (path, body) => {
  const response = await fetch(buildUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseJsonOrThrow(response, path);
};

export { API_BASE_URL };

// --- Live risk map (predicted_risk_class, depends on rainfall/water-level conditions) ---

// conditions: { model?, rainfall_24h_mm, water_level_m, river_station_id?, warning_level_m?, danger_level_m? }
// (rainfall_1h/3h/6h/12h_mm removed 2026-08 -- the model only uses rainfall_24h_mm now, see utils/risk.js toDynamicConditions)
export const getRiskMap = async (conditions) => requestJson("/api/risk-map", conditions);

// Same required params as getRiskMap. Returns a single dissolved Feature (or null geometry).
export const getAffectedArea = async (conditions) => requestJson("/api/risk-map/affected-area", conditions);

// Single-grid prediction, best model only.
// conditions: { rainfall_24h_mm, water_level_m }
export const predictGrid = async (gridId, conditions) =>
  postJson("/api/predict", { grid_id: gridId, conditions });

// Same request shape as predictGrid, runs every registered model (4 as of
// 2026-08: xgboost_tuned, xgboost, random_forest, logreg) -- response's
// `predictions` object keys itself off whatever the backend currently
// registers, so this comment's count is just informational, not load-bearing.
export const predictCompare = async (gridId, conditions) =>
  postJson("/api/predict/compare", { grid_id: gridId, conditions });

// Model metadata / accuracy for an about/methodology panel.
export const getModels = async () => requestJson("/api/models");

// Metadata about when/how the metrics returned by getModels() were computed
// (added 2026-08) -- a reproducible offline eval script (app/ml/evaluate.py),
// not a live computation, so these numbers only change when someone re-runs
// it after a model retrain. Response: { available, computed_at, dataset_path,
// dataset_row_count, train_row_count, val_row_count, test_row_count,
// test_rows_with_nan_river_level_score, river_level_score_fill_value,
// models_evaluated } when available=true, or { available: false, message }
// if the eval has never been run in this environment -- handle both.
export const getModelsEvaluationInfo = async () => requestJson("/api/models/evaluation-info");

// GIS output test results (proposal §3.13.2, added 2026-08) -- pre-computed
// (not live) comparison of the deployed model's predicted High/Danger zones
// against real historical flood-event records. Response has THREE figures
// under `results`, not one -- see `methodology_note` in the response and
// read it before displaying any single number:
//   - historical_evidence_override_only: the methodologically clean figure
//     (ground truth from independent outside evidence). Lead with this one.
//   - combined_all_historical_rows: favorable-looking but ~96% circular
//     (mostly reproduces the model's own training-label rule).
//   - reconstructed_only: the circular subset alone, shown for transparency
//     only, never as a headline claim.
// { available: false, message } if the test has never been run.
export const getGisOutputTestInfo = async () => requestJson("/api/models/gis-output-test");

// Real current rainfall (averaged across active Bharatpur-area BIPAD gauges)
// and river level (from the given station) — for pre-filling the conditions
// form / driving getRiskMap automatically. Throws (error.status === 502) if
// BIPAD is unreachable — callers must treat that as "unavailable," not "calm."
export const getLiveConditions = async (riverStationId = "devghat") =>
  requestJson("/api/conditions/live", { river_station_id: riverStationId });

// Static reference layer — Narayani/Rapti river corridor polygons. No params,
// doesn't change per request, fetch once and draw statically.
export const getRivers = async () => requestJson("/api/layers/rivers");

// Static reference layer — Bharatpur Metropolitan City's 29 ward boundaries
// (added 2026-08). Same shape/fetch-once-and-draw pattern as getRivers.
// Coordinate reprojection to WGS84 was empirically calibrated by the backend
// (source file had no declared CRS) -- if wards look offset from the risk
// grid once rendered, that's a backend data issue to flag back, not
// necessarily a frontend bug.
export const getWards = async () => requestJson("/api/layers/wards");

// Ward-wise risk report (added 2026-08) -- server-side aggregation, same
// required conditions params as getRiskMap (ward risk is just as condition-
// dependent as grid risk). Pre-sorted riskiest-first (`rank`). Distinguishes
// `wards_with_no_grid_coverage` (zero grids in the model's study-area
// population, e.g. wards far from the rivers) from wards that were scored
// and came back low-risk -- render those as "no data," never as "0% risk,"
// they're not the same claim.
export const getWardsRiskReport = async (conditions) => requestJson("/api/wards/risk-report", conditions);

// Forward-looking cumulative rainfall forecast (Open-Meteo, added 2026-08),
// for pre-filling Forecast Mode's rainfall_24h_mm field. horizon: "24h" |
// "48h" | "72h". Response: { horizon, rainfall_24h_mm, window_start,
// window_end, source: "open-meteo", fetched_at }. Throws (error.status ===
// 502) if Open-Meteo is unreachable — callers must treat that as
// "unavailable," never substitute a guessed value. Rainfall only: there is
// no river-level forecast source, so water_level_m still needs a live/
// manual value (see getLiveConditions).
export const getForecastConditions = async (horizon) =>
  requestJson("/api/conditions/forecast", { horizon });

// --- History (read-only; rows are written best-effort as a side effect of
// calls the app already makes — see app/routes/history.py. Not a guaranteed-
// complete audit trail.) ---

// Most recent single-grid predictions (written by predictGrid/POST /api/predict).
// Does NOT include predictCompare or getRiskMap runs — those aren't logged.
// gridId is optional — omit to see predictions across all grids.
export const getPredictionHistory = async (gridId, limit = 10) =>
  requestJson("/api/history/predictions", { grid_id: gridId, limit });

// Most recent rainfall readings, written by GET /api/conditions/live.
export const getRainfallHistory = async (limit = 50) =>
  requestJson("/api/history/rainfall", { limit });

// Most recent river-level readings, written by GET /api/conditions/live.
// stationId is optional — omit to see all stations.
export const getRiverLevelHistory = async (stationId, limit = 50) =>
  requestJson("/api/history/river-level", { station_id: stationId, limit });

// Most recent forecast fetches, written by GET /api/conditions/forecast.
// horizon is optional ("24h" | "48h" | "72h") — omit to see all horizons.
export const getForecastHistory = async (horizon, limit = 50) =>
  requestJson("/api/history/forecasts", { horizon, limit });

// Most recent High/Danger grids logged from GET /api/risk-map runs (added
// 2026-08) — doubles as this project's persisted alert history, since a
// RiskResultLog row only exists for alert-worthy (High/Danger) grids.
// Debounced server-side (at most one write per ~60s per conditions
// fingerprint), so this is "alert-worthy grids over time," not a complete
// per-request log of every risk-map call. gridId/runId are both optional
// filters; runId groups every row written by the same risk-map call.
export const getRiskResultHistory = async (gridId, runId, limit = 50) =>
  requestJson("/api/history/risk-results", { grid_id: gridId, run_id: runId, limit });

// --- Static / terrain-only views (no live prediction) ---

export const getAllGrids = async () => requestJson("/api/grids");

export const getGridById = async (gridId) => requestJson(`/api/grids/${gridId}`);

// Static susceptibility aggregate (terrain only, not live predicted risk).
export const getSummary = async () => requestJson("/api/summary");

export const getStaticRisk = async () => requestJson("/api/risk/static");

export const filterGrids = async (params = {}) => requestJson("/api/grids/filter", params);
