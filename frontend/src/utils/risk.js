// Exact color scheme from the flood risk spec — keep in sync with FloodMap/Legend/StatCards.
export const RISK_CLASS_ORDER = ["Low", "Medium", "High", "Danger"];

export const RISK_COLORS = {
  Low: "#2ecc71",
  Medium: "#f1c40f",
  High: "#e67e22",
  Danger: "#e74c3c",
};

export const MISSING_DATA_COLOR = "#b8bfc8";

export const getRiskColor = (riskClass) => RISK_COLORS[riskClass] ?? MISSING_DATA_COLOR;

// Proposal §1.5: "Alert and Recommendation Support: the system will provide
// flood alert information and basic recommendations based on the predicted
// risk level of selected locations." This is deliberately a static,
// deterministic lookup, not a model output -- the risk class is already the
// thing the ML model computed; turning it into plain-language guidance is a
// presentation concern the frontend can own outright, with no backend call
// needed. ALERT_CLASSES marks which classes actually warrant an alert badge
// (Low/Medium are informational, not alerts).
export const ALERT_CLASSES = ["High", "Danger"];

export const RISK_RECOMMENDATIONS = {
  Low: "No immediate action needed. Continue routine monitoring of rainfall and river conditions.",
  Medium: "Monitor rainfall and river levels closely over the next few hours. Avoid unnecessary travel near riverbanks or low-lying areas.",
  High: "Prepare to move valuables and vulnerable household members to higher ground. Avoid crossing flooded roads or riverbanks. Stay ready to evacuate on short notice.",
  Danger: "Evacuate immediately to designated safe or higher ground. Follow local disaster management office instructions. Avoid all riverbank and low-lying areas.",
};

export const getRiskRecommendation = (riskClass) => RISK_RECOMMENDATIONS[riskClass] ?? null;

// Manual-entry fallback used only when GET /api/conditions/live is
// unavailable (BIPAD unreachable) — deliberately an all-zero "dry day"
// rather than a guessed nonzero scenario, per the documented fallback
// behavior: this must read as "no data," not "conditions are calm."
//
// water_level_rv01_m/rv02_m (added 2026-08, backend river-station-resolution
// fix) are the two independent live readings — Narayani/Devghat and East
// Rapti/Rajaiya — and warning_level_rv01_m/rv02_m + danger_level_rv01_m/
// rv02_m are each station's own threshold pair (Devghat 7.3/9.0, Rajaiya
// 3.3/3.7 as of 2026-08 -- genuinely different numbers, not a typo).
// water_level_m/warning_level_m/danger_level_m are kept alongside them as
// DERIVED "currently active station" values (see getActiveStationWaterLevel/
// WarningLevel/DangerLevel below) — those are what POST /api/predict/
// predict-compare still take (single grid, single river, unaffected by the
// backend fix's shape) and what the Forecast/Simulation panels' single
// what-if sliders edit.
export const DEFAULT_CONDITIONS = {
  // xgboost_tuned (added 2026-08) is the backend's new best_model/default --
  // same trained XGBoost weights as plain xgboost, with a calibrated
  // Danger-classification threshold (higher precision, slightly lower
  // recall on Danger). See MODEL_OPTIONS/getModelOptions below.
  model: "xgboost_tuned",
  rainfall_24h_mm: 0,
  water_level_m: 0,
  water_level_rv01_m: 0,
  water_level_rv02_m: 0,
  river_station_id: "devghat",
  warning_level_m: "",
  danger_level_m: "",
  warning_level_rv01_m: "",
  danger_level_rv01_m: "",
  warning_level_rv02_m: "",
  danger_level_rv02_m: "",
};

// Static fallback list, used only before GET /api/models has loaded (or if
// it fails) -- once modelsData is available, getModelOptions/getModelLabel
// below read the live list instead, so a future 5th model or renamed
// display_name doesn't require a frontend code change to show up correctly.
// Kept in sync manually as a *fallback* only; xgboost_tuned added 2026-08
// when the backend made it the new best_model/default.
export const MODEL_OPTIONS = [
  { id: "xgboost_tuned", label: "XGBoost Fine-Tuned (best)" },
  { id: "xgboost", label: "XGBoost" },
  { id: "random_forest", label: "Random Forest" },
  { id: "logreg", label: "Logistic Regression" },
];

// Dropdown options for model-selector Selects (ConditionsForm, ForecastPanel).
// Prefers the live GET /api/models list (modelsData.models, each with
// model_id/display_name) so a model added/removed/renamed on the backend
// shows up automatically; falls back to the static MODEL_OPTIONS above only
// while modelsData hasn't loaded yet (or fetch failed) so the dropdown isn't
// briefly empty on mount.
export const getModelOptions = (modelsData) => {
  if (modelsData?.models?.length) {
    return modelsData.models.map((model) => ({
      id: model.model_id,
      label: model.model_id === modelsData.best_model ? `${model.display_name} (best)` : model.display_name,
    }));
  }
  return MODEL_OPTIONS;
};

// Display label for a single model id -- same live-first, static-fallback
// pattern as getModelOptions, for spots that show one model's name rather
// than a full dropdown (StatCards' "Active Model" card, the comparison
// drawer's per-card title). Consolidates what used to be three separately
// hardcoded MODEL_LABELS copies (StatCards.jsx, TransparencyCard.jsx,
// ModelComparePanel.jsx) that had already drifted out of sync once (none of
// them had xgboost_tuned) -- one source of truth now.
export const getModelLabel = (modelId, modelsData) => {
  const fromLive = modelsData?.models?.find((model) => model.model_id === modelId)?.display_name;
  if (fromLive) return fromLive;
  return MODEL_OPTIONS.find((option) => option.id === modelId)?.label?.replace(/ \(best\)$/, "") ?? modelId;
};

// The only two gauges GET /api/conditions/live accepts / that have known
// warning-danger thresholds server-side (app/ml/features.py).
export const RIVER_STATIONS = [
  { id: "devghat", label: "Devghat (Narayani)" },
  { id: "rajaiya", label: "Rajaiya (Rapti)" },
];

// Backend fix (2026-08): the static grid data resolves each grid's nearest
// gauge by CODE (RV01 = Narayani-side, RV02 = Rapti-side), previously never
// reconciled with the frontend's place-name station ids -- every grid used
// to get scored against whichever single station's threshold the frontend
// happened to send, misattributing a Narayani reading to Rapti-side grids
// and vice versa. Each grid now resolves its own station automatically
// server-side regardless of what the frontend sends. This map is only for
// the frontend's OWN bookkeeping (which of the two live readings feeds the
// single-grid predict/compare panel + which warning/danger pair to show).
const STATION_TO_RV_SUFFIX = { devghat: "rv01", rajaiya: "rv02" };

// Generic version of "whichever of the two per-river values matches the
// currently active gauge station" -- used for water_level, warning_level,
// and danger_level alike. baseKey + "_rv01_m"/"_rv02_m" must exist on
// conditions (see combineLiveConditions in App.jsx, which populates all
// three pairs from the two live fetches) for the correct per-station value
// to be found; falls back to the single non-suffixed `${baseKey}_m` field
// otherwise (e.g. a conditions object built before this field existed).
//
// A real bug this app shipped and had to fix: the gauge-station dropdown
// used to only recompute water_level_m on selection, never warning_level_m/
// danger_level_m -- so switching from Devghat to Rajaiya left the threshold
// fields frozen on whichever station was active when live data was last
// fetched (always Devghat by default), showing the same 7.3/9.0 for both
// rivers instead of Rajaiya's real 3.3/3.7. All three fields must be
// recomputed together on every station switch, which is why this is a
// single shared helper rather than three independent copies that could
// drift out of sync again.
const getActiveStationValue = (conditions, baseKey) => {
  const suffix = STATION_TO_RV_SUFFIX[conditions?.river_station_id] ?? STATION_TO_RV_SUFFIX.devghat;
  const rvValue = conditions?.[`${baseKey}_${suffix}_m`];
  if (rvValue !== undefined && rvValue !== null && rvValue !== "") {
    return Number(rvValue);
  }
  const fallback = conditions?.[`${baseKey}_m`];
  return fallback !== undefined && fallback !== null && fallback !== "" ? Number(fallback) : "";
};

export const getActiveStationWaterLevel = (conditions) => Number(getActiveStationValue(conditions, "water_level")) || 0;

// Same idea as getActiveStationWaterLevel, but for the warning/danger
// threshold pair -- these are optional overrides (can legitimately be "",
// meaning "let the backend auto-resolve"), so unlike water level this does
// NOT coerce a missing value to 0.
export const getActiveStationWarningLevel = (conditions) => getActiveStationValue(conditions, "warning_level");
export const getActiveStationDangerLevel = (conditions) => getActiveStationValue(conditions, "danger_level");

// Known published warning/danger thresholds for each gauge (app/ml/
// features.py RIVER_STATION_THRESHOLDS / app/services/bipad_client.py) --
// used ONLY as a last-resort fallback for scaling presets/scenario values
// between rivers when live threshold data hasn't loaded yet (a conditions
// object's danger_level_rv0Xm is still "" at that point, e.g. before the
// first GET /api/conditions/live resolves). The authoritative values always
// come from that live fetch, per station -- these are just so preset/
// scenario ratio-scaling doesn't break or silently use 0 before it does.
const FALLBACK_STATION_DANGER_LEVELS = { devghat: 9.0, rajaiya: 3.7 };

const getStationDangerLevel = (conditions, stationId) => {
  const suffix = STATION_TO_RV_SUFFIX[stationId] ?? STATION_TO_RV_SUFFIX.devghat;
  const live = conditions?.[`danger_level_${suffix}_m`];
  if (live !== undefined && live !== null && live !== "") {
    return Number(live);
  }
  return FALLBACK_STATION_DANGER_LEVELS[stationId] ?? FALLBACK_STATION_DANGER_LEVELS.devghat;
};

// Every quick-scenario preset (CONDITION_PRESETS/FORECAST_PRESETS below) and
// every scripted Simulation keyframe (utils/simulation.js) was calibrated
// against Narayani/Devghat's danger level (9.0m) -- "Extreme" = 9.6m, just
// past danger; "Dry" = 2m, well under warning; and so on. Applying that
// SAME absolute number to Rapti/Rajaiya (danger level 3.7m, well under
// half of Narayani's) breaks that calibration: 9.6m at Rapti is a
// physically implausible reading at ~2.6x ITS danger level, not "just past
// danger" the way it is for Narayani, and every preset in between has the
// same problem to a lesser degree -- both rivers rising in lockstep to the
// identical number doesn't reflect either river's real scale.
//
// This scales a Narayani-calibrated value down to the SAME proportion of
// Rapti's own danger level, preserving relative severity ("just past
// danger" reads as "just past danger" for both rivers) instead of an
// identical absolute number that ignores the ~2.4x difference between the
// two rivers' danger levels.
export const scaleWaterLevelToRapti = (narayaniWaterLevelM, conditions) => {
  const narayaniDanger = getStationDangerLevel(conditions, "devghat");
  const raptiDanger = getStationDangerLevel(conditions, "rajaiya");
  if (!narayaniDanger) return narayaniWaterLevelM;
  return Math.round(narayaniWaterLevelM * (raptiDanger / narayaniDanger) * 100) / 100;
};

// Applies a Narayani-calibrated water level to both rivers -- Narayani gets
// the value as-is, Rapti gets it ratio-scaled to its own danger level (see
// scaleWaterLevelToRapti) instead of the identical absolute number
// (the old withUniformWaterLevel behavior this replaces). Used by quick-
// scenario presets (ConditionsForm/ForecastPanel's applyPreset) and
// Simulation's scripted timeline (SimulationPanel's runStep/
// applyManualStep, when the user hasn't manually overridden a river).
// Also explicitly overwrites any stale water_level_rv01_m/rv02_m a caller
// might have carried over from spreading live conditions first, so
// toRiskMapParams doesn't silently keep sending old real per-river readings
// instead of the scenario's intended value (a real bug this app hit while
// building the two-river feature: presets/scenarios spread the current
// live conditions object first, which by then carries real rv01/rv02
// fields, then only overwrote a single water_level_m -- the stale rv01/rv02
// would otherwise win).
export const withRatioScaledWaterLevel = (conditions, narayaniWaterLevelM) => {
  const next = {
    ...conditions,
    water_level_rv01_m: narayaniWaterLevelM,
    water_level_rv02_m: scaleWaterLevelToRapti(narayaniWaterLevelM, conditions),
  };
  next.water_level_m = getActiveStationWaterLevel(next);
  return next;
};

// Slider ranges for the conditions form. rainfall_24h_mm is the ONLY
// rainfall window the model actually uses (app/ml/features.py, updated
// 2026-08 per PROJECT_STATE_UPDATE_SUMMARY.md -- the 1h/3h/6h/12h windows
// were dropped from training since they were silently 0.0 for a large
// share of training rows). The backend now ignores those fields entirely
// if sent, so there's no point exposing sliders for them here -- they'd
// visually imply a "what-if" effect on risk that doesn't actually happen.
// The max mirrors the model's own saturation point (rainfall_score pins at
// 1.0 beyond this). Water level tops out a bit above the highest known
// danger level (Devghat 9.0m).
export const RAINFALL_SLIDER_CONFIG = [{ key: "rainfall_24h_mm", label: "24h rainfall", max: 140 }];

export const WATER_LEVEL_SLIDER = { min: 0, max: 12, step: 0.1 };

// One-shot scenario presets for the conditions form (distinct from the
// multi-step, animated SIMULATION_SCENARIOS in utils/simulation.js) — pick
// one and it fills every slider at once for a quick "what if" test.
export const CONDITION_PRESETS = [
  {
    id: "dry",
    label: "Dry",
    color: "#2ecc71",
    values: { rainfall_24h_mm: 0, water_level_m: 2 },
  },
  {
    id: "light",
    label: "Light rain",
    color: "#9acd32",
    values: { rainfall_24h_mm: 30, water_level_m: 4 },
  },
  {
    id: "moderate",
    label: "Moderate storm",
    color: "#f1c40f",
    values: { rainfall_24h_mm: 75, water_level_m: 6 },
  },
  {
    id: "heavy",
    label: "Heavy storm",
    color: "#e67e22",
    values: { rainfall_24h_mm: 110, water_level_m: 7.8 },
  },
  {
    id: "extreme",
    label: "Extreme / cyclonic",
    color: "#e74c3c",
    values: { rainfall_24h_mm: 140, water_level_m: 9.6 },
  },
];

const safeNumber = (value) => {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
};

// SINGLE SOURCE OF TRUTH for the "dynamic conditions" payload sent to
// /api/predict, /api/predict/compare (as the `conditions` body field) and
// /api/risk-map / /api/risk-map/affected-area (as query params, alongside
// `model`). Every call site MUST build its payload through this function —
// do not hand-roll a second copy that "simplifies" the fields it sends.
//
// This existed as a bug before: a separate local helper for the
// predict/compare call site dropped river_station_id (and warning/danger
// levels), so the map (which sent the full conditions object) and the grid
// detail panel (which sent this trimmed copy) silently disagreed on the
// same grid at the same time — river_level_score fell back to a flat 0.3
// for the detail panel only, since river_station_id was missing. See
// diagnostics.river_level_score_used_fallback (predict/compare) / the
// risk-map response's grids_using_river_fallback count (added 2026-08,
// replacing an earlier top-level boolean of the same name), which the
// backend added specifically to catch this failure mode.
//
// rainfall_1h_mm/3h_mm/6h_mm/12h_mm are deliberately NOT sent: the backend
// (app/ml/features.py, updated 2026-08) only uses rainfall_24h_mm — those
// sub-hourly windows were dropped from training (silently 0.0 for a large
// share of training rows) and are no longer declared on /api/predict,
// /api/predict/compare, or /api/risk-map, so sending them would either be
// ignored (query params) or rejected as unrecognized (nothing to gain by
// keeping them in the payload).
export const toDynamicConditions = (conditions) => {
  const payload = {
    rainfall_24h_mm: Number(conditions.rainfall_24h_mm),
    water_level_m: Number(conditions.water_level_m),
  };

  if (conditions.river_station_id) {
    payload.river_station_id = conditions.river_station_id;
  }
  const warningLevel = safeNumber(conditions.warning_level_m);
  if (warningLevel !== null) {
    payload.warning_level_m = warningLevel;
  }
  const dangerLevel = safeNumber(conditions.danger_level_m);
  if (dangerLevel !== null) {
    payload.danger_level_m = dangerLevel;
  }

  return payload;
};

// Same fields as toDynamicConditions, plus the top-level `model` param
// /api/risk-map, /api/risk-map/affected-area and /api/wards/risk-report all
// take. Keeps the map's query params built from the exact same field
// selection as the detail panel's predict/compare body, instead of two
// independently-assembled objects that can drift.
//
// water_level_rv01_m/rv02_m (added 2026-08): the map endpoints score each
// grid against its OWN resolved river, so they take two independent
// readings instead of one uniform water_level_m. Always sent (never the
// bare water_level_m field) to avoid an ambiguous "which one wins" case on
// the backend -- if the caller's conditions object has real per-river
// values (ConditionsForm's dual sliders, live dual-fetch), those are used;
// otherwise both rivers get the same single water_level_m value (Forecast/
// Simulation's uniform what-if scenarios, which don't model two independent
// rivers, and old/legacy conditions objects that predate this field).
export const toRiskMapParams = (conditions) => {
  const { water_level_m, ...rest } = toDynamicConditions(conditions);
  const rv01 = safeNumber(conditions.water_level_rv01_m);
  const rv02 = safeNumber(conditions.water_level_rv02_m);
  return {
    model: conditions.model,
    ...rest,
    water_level_rv01_m: rv01 ?? water_level_m,
    water_level_rv02_m: rv02 ?? water_level_m,
  };
};

export const countByRiskClass = (features, key = "predicted_risk_class") => {
  const counts = { Low: 0, Medium: 0, High: 0, Danger: 0 };
  features.forEach((feature) => {
    const riskClass = feature?.properties?.[key];
    if (riskClass && Object.prototype.hasOwnProperty.call(counts, riskClass)) {
      counts[riskClass] += 1;
    }
  });
  return counts;
};

export const formatNumber = (value, digits = 2) => {
  const numericValue = safeNumber(value);
  if (numericValue === null) {
    return null;
  }

  return numericValue.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
};

// skipped_grids entries carry a free-text `reason` from the backend
// (app/ml/features.py) — grids below the trained-population area threshold
// say "sliver polygon" / "training-population threshold"; genuine data
// gaps say "missing required field". Bucket by that instead of treating
// every exclusion as a data-quality problem: most of them (133/141) are by
// design, not a gap.
export const isSliverExclusion = (reason) =>
  typeof reason === "string" && /sliver|training-population/i.test(reason);

export const bucketSkippedGrids = (skippedGrids) => {
  const slivers = [];
  const gaps = [];
  (skippedGrids ?? []).forEach((entry) => {
    if (isSliverExclusion(entry.reason)) {
      slivers.push(entry);
    } else {
      gaps.push(entry);
    }
  });
  return { slivers, gaps };
};

export const formatPercent = (value, digits = 1) => {
  const numericValue = safeNumber(value);
  if (numericValue === null) {
    return "N/A";
  }
  return `${(numericValue * 100).toFixed(digits)}%`;
};
