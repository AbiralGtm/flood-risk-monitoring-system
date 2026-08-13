import { getActiveStationWaterLevel } from "./risk";

// Forecast Risk Mode (proposal §3.2.2): lets a user generate a risk map for
// a named future horizon (24h/48h/72h) using forecast rainfall + an
// expected river level, instead of current/live conditions.
//
// IMPORTANT — be honest about what this actually is: the trained model has
// exactly one rainfall input, rainfall_24h_mm (see utils/risk.js
// toDynamicConditions / app/ml/features.py) — there is no separate 48h or
// 72h rainfall feature. So a "48h forecast" and a "72h forecast" only
// produce different results if you enter different rainfall/water-level
// numbers for them; the horizon itself is a label for your assumption, not
// a distinct signal the model reasons about. This mirrors the proposal's
// own framing: "the system will not forecast rainfall by itself... forecast
// rainfall will be used as an external input or entered manually" — there's
// no automated forecast feed wired in yet (see PROJECT_STATUS_REPORT.docx /
// the backend handoff prompt for what a real forecast feed would need).
export const FORECAST_HORIZONS = [
  { id: "24h", label: "+24 hours", hours: 24 },
  { id: "48h", label: "+48 hours", hours: 48 },
  { id: "72h", label: "+72 hours", hours: 72 },
];

export const DEFAULT_FORECAST_HORIZON = FORECAST_HORIZONS[0].id;

export const getHorizonLabel = (horizonId) =>
  FORECAST_HORIZONS.find((horizon) => horizon.id === horizonId)?.label ?? horizonId;

// Quick-start scenario presets for the forecast form. Same shape as
// utils/risk.js CONDITION_PRESETS (rainfall_24h_mm + water_level_m) — kept
// as a separate list (rather than reusing CONDITION_PRESETS directly) so
// forecast-specific wording/values can diverge later without touching the
// current-conditions presets.
export const FORECAST_PRESETS = [
  {
    id: "light",
    label: "Light rain expected",
    color: "#9acd32",
    values: { rainfall_24h_mm: 20, water_level_m: 4 },
  },
  {
    id: "moderate",
    label: "Moderate storm expected",
    color: "#f1c40f",
    values: { rainfall_24h_mm: 65, water_level_m: 6.2 },
  },
  {
    id: "heavy",
    label: "Heavy storm expected",
    color: "#e67e22",
    values: { rainfall_24h_mm: 100, water_level_m: 8 },
  },
  {
    id: "extreme",
    label: "Extreme event expected",
    color: "#e74c3c",
    values: { rainfall_24h_mm: 140, water_level_m: 9.6 },
  },
];

// Seed values for the forecast form. rainfall_24h_mm deliberately stays a
// fixed, obviously-a-placeholder light-rain guess (NOT live, NOT zero --
// zero would look like a finished, considered forecast rather than an empty
// form to fill in) since there's no forecasted-rainfall source to seed it
// from until the user hits "Fill from forecast."
//
// water_level_rv01_m/rv02_m are different: per SIMULATION_FORECAST_RIVER_
// FRONTEND_PROMPT.md, there's no river-level forecast source anywhere
// (Open-Meteo is rainfall-only; DHM/BIPAD don't publish one either), so
// "reuse the current live reading as your forecast starting point" is the
// most honest option -- these seed directly from baseConditions' own
// water_level_rv01_m/rv02_m (already live-fetched by Conditions/App.jsx,
// not a fresh API call from here) rather than a fixed placeholder, the same
// way Conditions itself pre-fills. Falls back to 4 only if baseConditions
// itself isn't available yet (mount race), not if it legitimately holds 0
// (0 means "live fetch failed, dry-day fallback" -- a real, meaningful
// value that should pass through as-is, which is why this uses `??` and
// not `||`).
//
// warning/danger_level_rv01_m/rv02_m are carried through too (not just the
// single active-station pair) so the forecast form's own gauge-station
// dropdown can re-derive the right threshold pair on switch, same as
// ConditionsForm -- see getActiveStationWarningLevel/DangerLevel in
// utils/risk.js and the bug they fixed (switching stations used to leave
// the threshold fields frozen on whichever was active when the form was
// seeded).
export const buildDefaultForecastConditions = (baseConditions) => {
  const seeded = {
    model: baseConditions?.model ?? "xgboost_tuned",
    rainfall_24h_mm: 20,
    water_level_rv01_m: baseConditions?.water_level_rv01_m ?? 4,
    water_level_rv02_m: baseConditions?.water_level_rv02_m ?? 4,
    river_station_id: baseConditions?.river_station_id ?? "devghat",
    warning_level_m: baseConditions?.warning_level_m ?? "",
    danger_level_m: baseConditions?.danger_level_m ?? "",
    warning_level_rv01_m: baseConditions?.warning_level_rv01_m ?? "",
    danger_level_rv01_m: baseConditions?.danger_level_rv01_m ?? "",
    warning_level_rv02_m: baseConditions?.warning_level_rv02_m ?? "",
    danger_level_rv02_m: baseConditions?.danger_level_rv02_m ?? "",
  };
  seeded.water_level_m = getActiveStationWaterLevel(seeded);
  return seeded;
};
