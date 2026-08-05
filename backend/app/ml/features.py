"""Builds the 14-column feature vector the current trained models expect,
from a grid's static properties plus live/forecast dynamic conditions.

Feature order (per the model's own saved `features` list -- see
app/ml/model_loader.py, which builds the inference DataFrame from
`bundle.features`, not a hardcoded list here):
    elevation_mean, slope_mean, distance_to_narayani_m, distance_to_rapti_m,
    nearest_river_distance, nearest_major_river_encoded, within_2km_narayani,
    within_2km_rapti, lulc_encoded, historical_flood_evidence,
    flow_accumulation_score, rainfall_24h_mm, rainfall_score,
    river_level_score

Updated 2026-08 per PROJECT_STATE_UPDATE_SUMMARY.md (ML/data pipeline
retrain). Changes from the previous 18-feature model:
  - rainfall_1h_mm/3h_mm/6h_mm/12h_mm and water_level_m are NO LONGER raw
    model features (the sub-hourly rainfall values were found to be
    silently 0.0 for a large share of training rows due to an upstream
    data-resolution mismatch; water_level_m was 38% NaN and redundant with
    river_level_score). water_level_m is still accepted as a *request*
    input, since it's still needed to compute river_level_score.
  - flow_accumulation_score (D8, via pysheds) is a new static feature.
  - rainfall_score is now derived from rainfall_24h_mm alone (confirmed
    with the user 2026-08-04, since the summary doc didn't restate this
    formula and the four sub-hourly windows it used to also incorporate
    are no longer trustworthy):
        rainfall_score = min(1.0, rainfall_24h_mm / 140)

    river_level_score: continuous ramp, see compute_river_level_score().
        fallback = 0.3 if no gauge/thresholds available (unchanged)

Note: the model was trained to predict risk_class directly from these
features, so no percentile-based risk_class thresholding is needed at
inference time -- that recalibration step only generates training labels
from final_risk_score during dataset build, not during live serving.
"""

from __future__ import annotations

from typing import Any

# LabelEncoder sorts classes alphabetically at fit time. "Narayani" < "Rapti"
# alphabetically, so this assumes narayani=0, rapti=1. VERIFY against the
# actual encoder/training script before relying on this for real predictions.
RIVER_ENCODING = {
    "narayani": 0,
    "rapti": 1,
}

# rainfall_score: single-window saturation on rainfall_24h_mm alone (the only
# rainfall window still considered reliable -- see module docstring). The
# 1h/3h/6h/12h saturation points this used to also take the max across are
# gone along with those windows as raw model inputs.
_RAINFALL_24H_SATURATION_MM = 140.0

RIVER_LEVEL_SCORE_FALLBACK = 0.3

# Per-station warning/danger levels (meters), used by compute_river_level_score.
# Sourced live from bipadportal.gov.np's public API (GET /api/v1/river/?search=<name>),
# which returns these as stable fields (warningLevel/dangerLevel) on every reading for
# a station -- confirmed identical across historical readings for each station, so
# they're the station's fixed thresholds, not a per-reading value:
#   "Narayani at Devghat": warningLevel=7.3, dangerLevel=9.0
#   "East Rapti at Rajaiya": warningLevel=3.3, dangerLevel=3.7
# Verified 2026-07-27 against live current data. Re-check periodically -- DHM does
# revise these occasionally after gauge recalibration.
#
# Two naming conventions exist for these same two stations and neither is
# going away: BIPAD's live API (app/services/bipad_client.py) identifies
# them by place name ("devghat"/"rajaiya"), while the static grid data's own
# river_station_id property (set by the data pipeline's
# 00_assign_stations_to_grid.py -- see data/static_grid_dataset_with_stations.geojson)
# uses gauge codes ("RV01"/"RV02"). These were never reconciled anywhere in
# the codebase, which silently broke river_level_score for any caller
# passing a grid's own river_station_id straight through (e.g. "RV01" never
# matched "devghat" in this table, so it silently fell through to the 0.3
# fallback -- confirmed via direct testing, identical output whether
# river_station_id was null or "RV01").
#
# Canonical on the gauge codes (they're what the static, per-grid data
# actually carries) and alias the place names to the same thresholds below,
# rather than maintaining two separate copies of the same numbers that
# could silently drift apart if DHM ever revises them.
_CANONICAL_STATION_THRESHOLDS: dict[str, dict[str, float]] = {
    "rv01": {"warning_level": 7.3, "danger_level": 9.0},  # Narayani at Devghat
    "rv02": {"warning_level": 3.3, "danger_level": 3.7},  # East Rapti at Rajaiya
}

_STATION_ALIASES: dict[str, str] = {
    "devghat": "rv01",
    "rajaiya": "rv02",
}

RIVER_STATION_THRESHOLDS: dict[str, dict[str, float]] = {
    **_CANONICAL_STATION_THRESHOLDS,
    **{alias: _CANONICAL_STATION_THRESHOLDS[code] for alias, code in _STATION_ALIASES.items()},
}


def station_id_aliases(station_id: str | None) -> list[str]:
    """All known spellings for the same physical gauge as `station_id` (e.g.
    "RV01" -> ["rv01", "devghat"]), lowercased.

    Needed anywhere that queries data logged under a *different* naming
    convention than a grid's own resolved station uses -- e.g. RiverLevelRecord
    rows are always logged with station_id "devghat"/"rajaiya" (see
    app/routes/conditions.py's GET /api/conditions/live, whose river_station_id
    query param only accepts those two spellings), but a grid's own resolved
    station (feature_row["_river_station_id"]) is normally "RV01"/"RV02" (the
    gauge code from the static grid data, not an explicit override). Querying
    RiverLevelRecord.station_id == "RV01" directly would silently return zero
    rows every time -- the exact class of naming-mismatch bug already fixed
    once for scoring (see RIVER_STATION_THRESHOLDS above); this is the same
    fix applied to history lookups instead.

    Returns [] if station_id is falsy. Returns [station_id.lower()] (just the
    input, lowercased) if it isn't a recognized station at all, so a caller
    can still filter on it without crashing.
    """
    if not station_id:
        return []
    key = station_id.strip().lower()
    thresholds = RIVER_STATION_THRESHOLDS.get(key)
    if thresholds is None:
        return [key]
    return [alias for alias, values in RIVER_STATION_THRESHOLDS.items() if values == thresholds]


def compute_rainfall_score(rainfall_24h_mm: float) -> float:
    return round(min(1.0, max(0.0, rainfall_24h_mm / _RAINFALL_24H_SATURATION_MM)), 6)


def compute_river_level_score(
    water_level_m: float,
    station_id: str | None = None,
    warning_level_m: float | None = None,
    danger_level_m: float | None = None,
) -> tuple[float, bool]:
    """warning_level_m/danger_level_m (if supplied directly) take priority
    over a station_id lookup in RIVER_STATION_THRESHOLDS. If neither
    resolves to a full pair of thresholds, returns the documented fallback.

    Continuous ramp per PROJECT_STATE_UPDATE_SUMMARY.md #3 (2026-08 pipeline
    update). This replaces the earlier two-segment formula, which returned a
    flat 0.3 for the entire 70%-100%-of-warning band -- that flat segment
    caused genuinely different river stages to score identically. Segments:
        stage < 0.7*warning        -> 0.1
        0.7*warning <= stage < warning  -> 0.1 to 0.3 (linear)
        warning <= stage < danger  -> 0.3 to 0.7 (linear)
        stage >= danger            -> 0.7 to 1.0 (linear, saturates at
                                       danger + 30% of danger over-threshold)

    Returns (score, used_fallback). used_fallback=True means no station
    thresholds could be resolved (e.g. river_station_id was omitted from
    the request) and water_level_m had NO effect on the score -- callers
    must surface this, not swallow it, since two requests that only differ
    in whether river_station_id was included can otherwise look like a
    prediction inconsistency when they're actually just differently-formed
    requests.
    """
    if warning_level_m is None or danger_level_m is None:
        station_thresholds = RIVER_STATION_THRESHOLDS.get((station_id or "").strip().lower())
        if station_thresholds:
            warning_level_m = warning_level_m if warning_level_m is not None else station_thresholds.get("warning_level")
            danger_level_m = danger_level_m if danger_level_m is not None else station_thresholds.get("danger_level")

    if warning_level_m is None or danger_level_m is None or warning_level_m <= 0 or danger_level_m <= warning_level_m:
        return RIVER_LEVEL_SCORE_FALLBACK, True

    low_bound = warning_level_m * 0.7

    if water_level_m < low_bound:
        score = 0.1
    elif water_level_m < warning_level_m:
        score = 0.1 + 0.2 * (water_level_m - low_bound) / (warning_level_m - low_bound)
    elif water_level_m < danger_level_m:
        score = 0.3 + 0.4 * (water_level_m - warning_level_m) / (danger_level_m - warning_level_m)
    else:
        excess = min((water_level_m - danger_level_m) / (danger_level_m * 0.3), 1.0)
        score = 0.7 + 0.3 * excess

    return round(score, 6), False


# Matches the training pipeline's population filter exactly (03_build_ml_training_dataset.py:
# df = df[df["area_km2"] >= 0.005]). Sliver polygons left over from river-overlap correction
# were never part of the trained population, and most of them also fail zonal-stats computation
# (null slope_mean etc) -- but not all of them, and a handful of grids fail zonal stats without
# being slivers. Checking area first and independently from the required-field check is what
# correctly separates "not in the training population" (133 grids) from "in the population but
# missing data" (8 genuine gaps, e.g. GRID_1896) rather than conflating them under one count.
MIN_GRID_AREA_KM2 = 0.005


def _check_training_population(properties: dict[str, Any]) -> None:
    """Raise ValueError if this grid falls outside the model's trained population
    (area < MIN_GRID_AREA_KM2 km², i.e. a sliver polygon), before checking for
    missing required fields.
    """
    area_km2 = properties.get("area_km2")
    if area_km2 is None:
        return  # no area recorded -- let the required-field checks catch this grid instead
    if area_km2 < MIN_GRID_AREA_KM2:
        grid_id = properties.get("grid_id", "<unknown>")
        raise ValueError(
            f"Grid '{grid_id}' area ({area_km2} km²) is below the {MIN_GRID_AREA_KM2} km² "
            "training-population threshold (sliver polygon) -- not part of the model's "
            "trained population, excluded independently of any missing-field check."
        )


def _required_float(properties: dict[str, Any], key: str) -> float:
    """float() a required grid property, raising a clear, catchable error
    (rather than a bare TypeError) if it's missing or null in the source data.
    """
    value = properties.get(key)
    if value is None:
        grid_id = properties.get("grid_id", "<unknown>")
        raise ValueError(f"Grid '{grid_id}' is missing required field '{key}' (null in source data).")
    return float(value)


def encode_nearest_major_river(river_name: str | None) -> int:
    if not river_name:
        return -1
    key = river_name.strip().lower()
    if key not in RIVER_ENCODING:
        raise ValueError(f"Unknown nearest_major_river value '{river_name}'.")
    return RIVER_ENCODING[key]


def build_feature_row(
    grid_properties: dict[str, Any],
    dynamic_conditions: dict[str, Any],
) -> dict[str, float]:
    """Assemble one feature row combining a grid's static properties with
    live/forecast dynamic conditions supplied by the caller.

    grid_properties: a feature's `properties` dict from the static grid
        GeoJSON. Expected keys: elevation_mean, slope_mean,
        distance_to_narayani_m, distance_to_rapti_m, nearest_river_distance,
        nearest_major_river, within_2km_narayani, within_2km_rapti,
        lulc_encoded, flow_accumulation_score, and optionally
        historical_flood_evidence, river_station_id (used as a fallback --
        see below -- when the request doesn't specify one; "RV01"/"RV02" in
        the current dataset, aliased to "devghat"/"rajaiya" in
        RIVER_STATION_THRESHOLDS above).

        flow_accumulation_score currently has NO source in the deployed
        static grid file (data/bharatpur_static_grid_250m.geojson) -- it's a
        new column from the 2026-08 pipeline update that hasn't been wired
        in yet (see PROJECT_STATE_UPDATE_SUMMARY.md #2, and
        app/config.py:DATA_PATH). Until the canonical
        static_grid_dataset_with_stations file replaces it, every grid will
        fail here with a clear ValueError rather than silently defaulting to
        0 -- a wrong default would silently feed the model an incorrect
        value on every single prediction, exactly the class of bug this
        pipeline update is trying to eliminate.
    dynamic_conditions: request-supplied live/forecast values. Expected
        keys: rainfall_24h_mm, water_level_m (used only to derive
        river_level_score, not passed to the model directly -- see module
        docstring), and optionally river_station_id, warning_level_m,
        danger_level_m.
    """
    _check_training_population(grid_properties)

    distance_to_narayani_m = _required_float(grid_properties, "distance_to_narayani_m")
    distance_to_rapti_m = _required_float(grid_properties, "distance_to_rapti_m")
    within_2km_narayani = grid_properties["within_2km_narayani"]
    within_2km_rapti = grid_properties["within_2km_rapti"]

    rainfall_24h_mm = float(dynamic_conditions["rainfall_24h_mm"])
    water_level_m = float(dynamic_conditions["water_level_m"])

    rainfall_score = compute_rainfall_score(rainfall_24h_mm)

    # Prefer an explicit request override, but fall back to the grid's own
    # pre-assigned river_station_id (set once in the data pipeline based on
    # real geography -- see 00_assign_stations_to_grid.py, and
    # RIVER_STATION_THRESHOLDS above for why both "RV01"/"RV02" and
    # "devghat"/"rajaiya" now resolve identically) rather than silently
    # falling through to the 0.3 fallback whenever the caller doesn't
    # supply one. This is what lets a caller omit river_station_id entirely
    # and still get a geographically-correct score, instead of requiring
    # every caller to already know and correctly pass each grid's river.
    station_id = dynamic_conditions.get("river_station_id") or grid_properties.get("river_station_id")

    river_level_score, river_level_score_used_fallback = compute_river_level_score(
        water_level_m,
        station_id=station_id,
        warning_level_m=dynamic_conditions.get("warning_level_m"),
        danger_level_m=dynamic_conditions.get("danger_level_m"),
    )

    return {
        "elevation_mean": _required_float(grid_properties, "elevation_mean"),
        "slope_mean": _required_float(grid_properties, "slope_mean"),
        "distance_to_narayani_m": distance_to_narayani_m,
        "distance_to_rapti_m": distance_to_rapti_m,
        "nearest_river_distance": _required_float(grid_properties, "nearest_river_distance"),
        "nearest_major_river_encoded": encode_nearest_major_river(
            grid_properties.get("nearest_major_river")
        ),
        "within_2km_narayani": int(bool(within_2km_narayani)),
        "within_2km_rapti": int(bool(within_2km_rapti)),
        "lulc_encoded": _required_float(grid_properties, "lulc_encoded"),
        "historical_flood_evidence": float(grid_properties.get("historical_flood_evidence", 0)),
        "flow_accumulation_score": _required_float(grid_properties, "flow_accumulation_score"),
        "rainfall_24h_mm": rainfall_24h_mm,
        "rainfall_score": rainfall_score,
        "river_level_score": river_level_score,
        # Diagnostic-only fields, not model inputs (bundle.predict_batch only reads columns
        # named in the model's own `features` list, so these are safely ignored at inference
        # time). Callers should surface these rather than discard them -- see the map-vs-detail-
        # panel inconsistency this was added to prevent from recurring silently.
        "_river_level_score_used_fallback": river_level_score_used_fallback,
        # The *resolved* station (request override or grid's own static
        # value), not just the raw request input -- otherwise this would
        # keep showing null even when the grid's own value was successfully
        # used, which would look like the fallback bug is still present
        # when it isn't.
        "_river_station_id": station_id,
    }
