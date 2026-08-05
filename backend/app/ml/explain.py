"""Rule-based contributing-factor explanations for a single prediction
(proposal Section 3.7):

    "Rule-based explanation may also be used to describe the major
    contributing factors behind the predicted risk, such as heavy rainfall,
    proximity to river, low relative elevation, high flow accumulation, or
    historical flood evidence."

Deterministic thresholds, not SHAP/LIME-style statistical attribution --
matching the proposal's own "rule-based" language and its five named
example factor types (plus river/water level, a sixth that isn't in the
proposal's named list but is an obviously major real driver, since rainfall
and river level are the model's only two live-condition inputs).

Single source of truth: this takes the exact feature_row build_feature_row()
already produced for the prediction being explained, so the explanation
always describes the same values the model actually scored -- never a
second, independently-recomputed set of numbers that can silently drift
from what was fed to the model (see app/routes/risk_map.py's map-vs-detail-
panel bug history -- that bug happened exactly this way, two code paths
computing "the same" value differently).
"""

from __future__ import annotations

from typing import Any

from app.ml.features import RIVER_ENCODING

# Reverse of RIVER_ENCODING (app/ml/features.py) so factor labels can name
# the actual nearest river ("Narayani"/"Rapti") from
# nearest_major_river_encoded alone, without needing the raw grid
# properties dict as a second input.
_RIVER_NAMES_BY_CODE = {code: name.capitalize() for name, code in RIVER_ENCODING.items()}

# Thresholds below are calibrated against the current 2,434-grid canonical
# dataset (data/static_grid_dataset_with_stations.geojson), checked directly
# (min/max/mean/percentiles) on 2026-08 -- not eyeballed, and not the
# proposal/earlier prompt's 2,575-grid-era estimates. Re-check these if the
# canonical dataset is ever regenerated with materially different grids.
#
#   elevation_mean:          min 142.45m, max 300.40m, mean 178.67m, median 174.83m
#   flow_accumulation_score: right-skewed 0-1, median 0.047, p75=0.1265, p90=0.3228
#   nearest_river_distance:  min 0m, max 1986.3m, mean 977.7m

_RAINFALL_HIGH = 0.8
_RAINFALL_MODERATE = 0.5
_RAINFALL_LOW = 0.25

_RIVER_LEVEL_HIGH = 0.7
_RIVER_LEVEL_MODERATE = 0.3

_RIVER_DISTANCE_HIGH_M = 200.0
_RIVER_DISTANCE_MODERATE_M = 500.0

# elevation_mean here, not a "relative" elevation -- the proposal's own
# Table 3.2 draft lists "relative elevation" as a feature, but no per-grid
# normalized/relative elevation field exists in the current dataset or the
# deployed model's feature set (see app/ml/features.py's build_feature_row
# docstring). Using absolute elevation_mean instead is a deliberate,
# documented gap, not an oversight.
_ELEVATION_HIGH_M = 150.0
_ELEVATION_MODERATE_M = 160.0

# ~p90 and ~p75 of the current dataset's flow_accumulation_score distribution.
_FLOW_HIGH = 0.3
_FLOW_MODERATE = 0.13

_SEVERITY_ORDER = {"high": 0, "moderate": 1, "low": 2}


def _nearest_river_name(feature_row: dict[str, Any]) -> str:
    code = feature_row.get("nearest_major_river_encoded")
    return _RIVER_NAMES_BY_CODE.get(code, "a nearby river")


def compute_contributing_factors(
    feature_row: dict[str, Any], diagnostics: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    """Returns entries like:
        {"factor": "rainfall", "label": "Very heavy rainfall (145mm in 24h)", "severity": "high"}
    ordered by severity (high -> moderate -> low), most severe first. Each
    rule is independent -- a grid can trigger several at once -- and only
    rules that actually trigger appear, so a Low-risk grid under calm
    conditions can return an empty list rather than padding it out.

    `feature_row` must be the same dict build_feature_row() produced for
    this prediction (including its underscore-prefixed diagnostic fields,
    e.g. _river_level_score_used_fallback). `diagnostics` is optional and
    only needed if the caller already has a separately-shaped diagnostics
    dict (e.g. app/routes/predict.py's _diagnostics()) it would rather pass
    instead of re-reading feature_row's underscore-prefixed fields directly
    -- both are read from the same underlying values either way.
    """
    factors: list[dict[str, str]] = []

    def _used_fallback() -> bool:
        if diagnostics is not None and "river_level_score_used_fallback" in diagnostics:
            return bool(diagnostics["river_level_score_used_fallback"])
        return bool(feature_row.get("_river_level_score_used_fallback", False))

    # 1. Rainfall
    rainfall_24h_mm = feature_row.get("rainfall_24h_mm")
    rainfall_score = feature_row.get("rainfall_score")
    if rainfall_24h_mm is not None and rainfall_score is not None:
        if rainfall_score >= _RAINFALL_HIGH:
            factors.append(
                {
                    "factor": "rainfall",
                    "label": f"Very heavy rainfall ({rainfall_24h_mm:.0f}mm in 24h)",
                    "severity": "high",
                }
            )
        elif rainfall_score >= _RAINFALL_MODERATE:
            factors.append(
                {
                    "factor": "rainfall",
                    "label": f"Heavy rainfall ({rainfall_24h_mm:.0f}mm in 24h)",
                    "severity": "moderate",
                }
            )
        elif rainfall_score >= _RAINFALL_LOW:
            factors.append(
                {
                    "factor": "rainfall",
                    "label": f"Elevated rainfall ({rainfall_24h_mm:.0f}mm in 24h)",
                    "severity": "low",
                }
            )

    # 2. River / water level -- a live-condition factor, distinct from #3's
    # static river-proximity factor. Suppressed entirely (not just caveated)
    # when no real gauge/threshold data was resolved: the score is a generic
    # 0.3 fallback in that case, not a real reading, so attributing risk to
    # it would misrepresent an estimate as evidence.
    if not _used_fallback():
        river_level_score = feature_row.get("river_level_score")
        if river_level_score is not None:
            if river_level_score >= _RIVER_LEVEL_HIGH:
                factors.append(
                    {
                        "factor": "river_level",
                        "label": "River level at or above the danger threshold",
                        "severity": "high",
                    }
                )
            elif river_level_score >= _RIVER_LEVEL_MODERATE:
                factors.append(
                    {
                        "factor": "river_level",
                        "label": "River level approaching the warning threshold",
                        "severity": "moderate",
                    }
                )

    # 3. Proximity to river (static/terrain factor, distinct from #2's live
    # water level -- a grid can be right next to the river while the river
    # itself is calm, or far from it during a genuine gauge-confirmed surge).
    nearest_river_distance = feature_row.get("nearest_river_distance")
    if nearest_river_distance is not None and nearest_river_distance < _RIVER_DISTANCE_HIGH_M:
        factors.append(
            {
                "factor": "river_proximity",
                "label": f"Very close to the {_nearest_river_name(feature_row)} ({nearest_river_distance:.0f}m)",
                "severity": "high",
            }
        )
    elif nearest_river_distance is not None and nearest_river_distance < _RIVER_DISTANCE_MODERATE_M:
        factors.append(
            {
                "factor": "river_proximity",
                "label": f"Close to the {_nearest_river_name(feature_row)} ({nearest_river_distance:.0f}m)",
                "severity": "moderate",
            }
        )
    elif feature_row.get("within_2km_narayani") or feature_row.get("within_2km_rapti"):
        factors.append(
            {"factor": "river_proximity", "label": "Within 2km of a major river", "severity": "low"}
        )

    # 4. Low (absolute) elevation
    elevation_mean = feature_row.get("elevation_mean")
    if elevation_mean is not None and elevation_mean < _ELEVATION_HIGH_M:
        factors.append(
            {
                "factor": "low_elevation",
                "label": f"Very low-lying area (elevation {elevation_mean:.0f}m)",
                "severity": "high",
            }
        )
    elif elevation_mean is not None and elevation_mean < _ELEVATION_MODERATE_M:
        factors.append(
            {
                "factor": "low_elevation",
                "label": f"Low-lying area (elevation {elevation_mean:.0f}m)",
                "severity": "moderate",
            }
        )

    # 5. High flow accumulation
    flow_accumulation_score = feature_row.get("flow_accumulation_score")
    if flow_accumulation_score is not None and flow_accumulation_score >= _FLOW_HIGH:
        factors.append(
            {
                "factor": "flow_accumulation",
                "label": "Very high water-flow convergence at this location",
                "severity": "high",
            }
        )
    elif flow_accumulation_score is not None and flow_accumulation_score >= _FLOW_MODERATE:
        factors.append(
            {
                "factor": "flow_accumulation",
                "label": "High water-flow convergence at this location",
                "severity": "moderate",
            }
        )

    # 6. Historical flood evidence -- trained feature importance is
    # documented elsewhere as low (~0.003), but still worth surfacing: it's
    # independently meaningful evidence a human reader would want to know
    # about even though the model itself barely weighs it.
    if feature_row.get("historical_flood_evidence") == 1:
        factors.append(
            {
                "factor": "historical_flood_evidence",
                "label": "This location has recorded historical flood evidence",
                "severity": "high",
            }
        )

    factors.sort(key=lambda f: _SEVERITY_ORDER.get(f["severity"], 99))
    return factors
