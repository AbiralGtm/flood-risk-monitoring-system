"""Static-only susceptibility scoring (no dynamic rainfall/river conditions).

Implements the documented formula (updated 2026-08 per
PROJECT_STATE_UPDATE_SUMMARY.md #2, which added flow_accumulation_score and
reduced slope_score's weight to make room for it):

    static_susceptibility_score =
        0.30*river_distance_score + 0.25*elevation_score +
        0.15*slope_score + 0.15*lulc_score + 0.10*history_score +
        0.05*flow_accumulation_score

This is NOT the trained model's predicted_risk_class -- that requires
dynamic conditions and comes from /api/predict. This is a static-only view
used for /api/risk/static and /api/summary, so the map/dashboard has
something to show before a live rainfall/river reading is supplied.

history_score and flow_accumulation_score both default to 0 for every grid
right now: the currently-deployed static dataset (bharatpur_static_grid_250m.geojson)
does not carry either field. flow_accumulation_score in particular is a new
column from the 2026-08 pipeline update that requires switching to the
canonical static_grid_dataset_with_stations file (see app/config.py:DATA_PATH,
not yet done -- pending that file). Unlike the live model's feature vector
(app/ml/features.py), a 0-default here only affects this static-only
dashboard view, not what the trained model actually sees, so it's an
acceptable placeholder until the new file is wired in -- but the static
susceptibility numbers shown today do NOT yet reflect flow accumulation.
"""

from __future__ import annotations

from typing import Any

from app.utils import get_geojson_data

_WEIGHTS = {
    "river_distance_score": 0.30,
    "elevation_score": 0.25,
    "slope_score": 0.15,
    "lulc_score": 0.15,
    "history_score": 0.10,
    "flow_accumulation_score": 0.05,
}

# Static-only view uses the same percentile cut points the training
# pipeline uses for dynamic risk_class (70th/90th/98th), applied to the
# static_susceptibility_score distribution across all grids.
_PERCENTILE_CUTS = (0.70, 0.90, 0.98)
_CLASS_LABELS = ("Low", "Medium", "High", "Danger")


def compute_static_susceptibility_score(properties: dict[str, Any]) -> float:
    history_score = float(properties.get("history_score", 0) or 0)
    flow_accumulation_score = float(properties.get("flow_accumulation_score", 0) or 0)
    score = (
        _WEIGHTS["river_distance_score"] * float(properties.get("river_distance_score", 0) or 0)
        + _WEIGHTS["elevation_score"] * float(properties.get("elevation_score", 0) or 0)
        + _WEIGHTS["slope_score"] * float(properties.get("slope_score", 0) or 0)
        + _WEIGHTS["lulc_score"] * float(properties.get("lulc_score", 0) or 0)
        + _WEIGHTS["history_score"] * history_score
        + _WEIGHTS["flow_accumulation_score"] * flow_accumulation_score
    )
    return round(score, 6)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _class_thresholds(scores: list[float]) -> tuple[float, float, float]:
    ordered = sorted(scores)
    return tuple(_percentile(ordered, cut) for cut in _PERCENTILE_CUTS)  # type: ignore[return-value]


def _classify(score: float, thresholds: tuple[float, float, float]) -> str:
    low_cut, medium_cut, high_cut = thresholds
    if score >= high_cut:
        return "Danger"
    if score >= medium_cut:
        return "High"
    if score >= low_cut:
        return "Medium"
    return "Low"


def get_static_risk_scores() -> list[dict[str, Any]]:
    """Return [{grid_id, static_susceptibility_score, static_susceptibility_class}, ...]."""
    features = get_geojson_data()["features"]
    scored = []
    for feature in features:
        properties = feature.get("properties", {})
        score = compute_static_susceptibility_score(properties)
        scored.append({"grid_id": properties.get("grid_id"), "score": score})

    thresholds = _class_thresholds([s["score"] for s in scored])
    for entry in scored:
        entry["static_susceptibility_class"] = _classify(entry["score"], thresholds)

    return scored


def static_risk_class_counts() -> dict[str, int]:
    counts: dict[str, int] = {label: 0 for label in _CLASS_LABELS}
    for entry in get_static_risk_scores():
        counts[entry["static_susceptibility_class"]] += 1
    return counts
