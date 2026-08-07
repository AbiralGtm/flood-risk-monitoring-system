import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.models import RiskResultLog
from app.db.session import get_db
from app.ml.features import build_feature_row
from app.ml.model_loader import BEST_MODEL_ID, get_model, list_model_ids
from app.utils import build_feature_collection, get_geojson_data

router = APIRouter()

# Debounce for RiskResultLog writes (see app/db/models.py's docstring):
# /api/risk-map gets hit on every interactive map refresh, and logging every
# High/Danger grid on every one of those would spam the table. This is an
# in-memory, per-process cache -- same pattern as app/services/bipad_client.py's
# _CACHE_TTL_SECONDS/_cache -- keyed on a rounded conditions fingerprint so
# near-identical repeated requests (e.g. the same preset clicked twice) don't
# re-log. Resets on restart; that's fine, it only affects debounce timing, not
# correctness of what gets logged.
_DEBOUNCE_SECONDS = 60
_last_logged_at: dict[tuple[Any, ...], float] = {}


def _round_or_none(value: Any) -> float | None:
    return round(float(value), 1) if value is not None else None


def _conditions_fingerprint(model_id: str, conditions: dict[str, Any]) -> tuple[Any, ...]:
    """Debounce key for RiskResultLog writes. Includes the legacy global
    water_level_m AND both per-station readings (see Fix 3 /
    RIVER_STATION_RESOLUTION_BACKEND_PROMPT.md) -- water_level_m alone is no
    longer guaranteed to be present (a caller can supply only per-station
    values now), and two requests that differ only in one station's reading
    must not be treated as the same fingerprint.
    """
    water_level_by_station = conditions.get("water_level_by_station") or {}
    return (
        model_id,
        round(float(conditions["rainfall_24h_mm"]), 1),
        _round_or_none(conditions.get("water_level_m")),
        _round_or_none(water_level_by_station.get("rv01")),
        _round_or_none(water_level_by_station.get("rv02")),
    )


def _log_risk_result_if_due(
    db: Session, model_id: str, conditions: dict[str, Any], enriched_features: list[dict[str, Any]]
) -> None:
    """Best-effort, debounced history write (see app/db/models.py RiskResultLog
    docstring). A DB hiccup or skipped debounce window must never affect the
    /api/risk-map response itself."""
    fingerprint = _conditions_fingerprint(model_id, conditions)
    now = time.monotonic()
    last_logged = _last_logged_at.get(fingerprint)
    if last_logged is not None and (now - last_logged) < _DEBOUNCE_SECONDS:
        return

    alert_features = [
        f for f in enriched_features if f["properties"]["predicted_risk_class"] in ("High", "Danger")
    ]
    if not alert_features:
        # Nothing alert-worthy this run -- still mark the fingerprint as "seen" so a
        # string of all-Low/Medium refreshes doesn't retry a DB write every single call.
        _last_logged_at[fingerprint] = now
        return

    run_id = str(uuid.uuid4())
    try:
        for feature in alert_features:
            properties = feature["properties"]
            db.add(
                RiskResultLog(
                    grid_id=str(properties.get("grid_id")),
                    model_id=model_id,
                    predicted_risk_class=properties["predicted_risk_class"],
                    rainfall_24h_mm=float(conditions["rainfall_24h_mm"]),
                    # This grid's own resolved water level (set in
                    # score_all_grids -- see Fix 3), not a single global
                    # conditions["water_level_m"] -- grids on different
                    # rivers can now have genuinely different readings in
                    # one request, so logging the global value would
                    # misrepresent what this specific grid was actually
                    # scored with.
                    water_level_m=float(properties.get("water_level_m") or 0.0),
                    run_id=run_id,
                )
            )
        db.commit()
        _last_logged_at[fingerprint] = now
    except Exception:  # noqa: BLE001 - logging is best-effort, never fail the request over it
        db.rollback()


def _resolve_grid_conditions(properties: dict[str, Any], conditions: dict[str, Any]) -> dict[str, Any]:
    """Resolve ONE grid's own water_level_m/river_station_id from the
    request's conditions (see RIVER_STATION_RESOLUTION_BACKEND_PROMPT.md
    Fix 3). Every grid has its own real river_station_id ("RV01"/"RV02" --
    Narayani/Rapti, see app/config.py's DATA_PATH-adjacent static data), so
    a single global water_level_m applied to all 2,434 grids made it
    architecturally impossible to represent "Narayani rising, Rapti
    constant" -- a rising Narayani reading bled into every Rapti-side
    grid's river_level_score too.

    Water level resolution, per grid:
        1. `water_level_by_station[grid's own station]`, if the caller
           supplied a true per-station reading for it.
        2. Else the legacy single `water_level_m`, applied uniformly
           (deprecated but kept working -- see condition_params()).
        3. Else 0.0 -- compute_river_level_score() turns this into a real
           (low) score using the grid's own resolved thresholds below, not
           the generic 0.3 fallback, since a valid station_id is still
           available even when no reading was supplied for it.

    Station resolution, per grid: unlike single-grid /api/predict (which
    lets an explicit request override win -- a reasonable single-grid
    what-if query), the bulk map deliberately does NOT let a request-level
    river_station_id force every grid onto one station's thresholds --
    that's precisely the bug this fix removes. Each grid always resolves
    to its own real, pre-assigned station. The legacy `river_station_id`
    query param is still accepted (so existing callers don't 422), it just
    no longer does anything at this endpoint -- see condition_params()'s
    docstring.
    """
    grid_station = (properties.get("river_station_id") or "").strip().lower()
    water_level_by_station = conditions.get("water_level_by_station") or {}
    resolved_water_level = water_level_by_station.get(grid_station)
    if resolved_water_level is None:
        resolved_water_level = conditions.get("water_level_m")

    return {
        **conditions,
        "water_level_m": resolved_water_level if resolved_water_level is not None else 0.0,
        "river_station_id": properties.get("river_station_id"),
    }


def score_all_grids(
    model_id: str, conditions: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Score every grid, each resolving its own water_level_m/river_station_id
    from `conditions` (see _resolve_grid_conditions -- Fix 3 of
    RIVER_STATION_RESOLUTION_BACKEND_PROMPT.md; grids are no longer forced
    to share one global reading regardless of which river they're near).

    Returns (enriched_features, skipped) -- grids whose static properties
    are missing a required field (data-quality gaps, not code bugs) are
    skipped rather than failing the whole map, and reported in `skipped`
    so the gap is visible instead of silently dropped.

    Shared across routes (not just /api/risk-map): also used by
    app/routes/wards.py's ward-wise risk report, which needs the identical
    per-grid scoring before aggregating by ward.
    """
    if model_id not in list_model_ids():
        raise HTTPException(
            status_code=404,
            detail={"error": "Unknown model", "message": f"No model '{model_id}'"},
        )

    features = get_geojson_data()["features"]

    feature_rows = []
    scoreable_features = []
    grid_conditions_list: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for feature in features:
        properties = feature.get("properties", {})
        grid_conditions = _resolve_grid_conditions(properties, conditions)
        try:
            feature_rows.append(build_feature_row(properties, grid_conditions))
            scoreable_features.append(feature)
            grid_conditions_list.append(grid_conditions)
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=501, detail={"error": "Not implemented", "message": str(exc)}
            ) from exc
        except (KeyError, ValueError) as exc:
            skipped.append({"grid_id": str(properties.get("grid_id")), "reason": str(exc)})

    bundle = get_model(model_id)
    try:
        predictions = bundle.predict_batch(feature_rows)
    except Exception as exc:  # noqa: BLE001 - e.g. a feature the model needs isn't in every
        # row build_feature_row produced (a model/feature-builder mismatch, not a per-grid
        # data gap -- those are already caught above). Report clearly instead of a raw 500.
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Batch prediction failed",
                "message": str(exc),
                "hint": (
                    "This usually means the model's saved feature list and "
                    "build_feature_row() are out of sync -- e.g. the model was "
                    "retrained with a new/changed feature set that the feature "
                    "builder doesn't produce yet."
                ),
            },
        ) from exc

    enriched_features = []
    for feature, prediction, feature_row, grid_conditions in zip(
        scoreable_features, predictions, feature_rows, grid_conditions_list
    ):
        properties = dict(feature.get("properties", {}))
        properties["predicted_risk_class"] = prediction["predicted_risk_class"]
        properties["risk_probability"] = prediction["risk_probability"]
        properties["risk_probabilities"] = prediction["probabilities"]
        properties["model_id"] = model_id
        # Surfaced so a grid that looks "wrong" can be diagnosed from the map response
        # alone -- see the map-vs-detail-panel inconsistency this was added to catch.
        properties["river_level_score"] = feature_row["river_level_score"]
        properties["river_level_score_used_fallback"] = feature_row["_river_level_score_used_fallback"]
        # This grid's own resolved reading/station (Fix 3) -- can genuinely
        # differ grid-to-grid now, unlike everything above which was always
        # identical across the whole map before this fix.
        properties["water_level_m"] = grid_conditions["water_level_m"]
        properties["river_station_id"] = feature_row["_river_station_id"]
        enriched_features.append({**feature, "properties": properties})

    return enriched_features, skipped


def condition_params(
    # rainfall_1h_mm/3h_mm/6h_mm/12h_mm removed 2026-08 -- no longer used by
    # the model (see app/ml/features.py module docstring). If the frontend
    # still sends them as query params they're simply ignored by FastAPI
    # (undeclared query params don't cause a validation error), so this is
    # not a breaking change for existing callers.
    rainfall_24h_mm: float = Query(..., ge=0),
    # water_level_m is now OPTIONAL and DEPRECATED at this endpoint (was
    # previously required) -- see RIVER_STATION_RESOLUTION_BACKEND_PROMPT.md
    # Fix 3. Existing callers that only pass this keep working exactly as
    # before (applied uniformly to whichever station each grid resolves
    # to); new callers should prefer water_level_rv01_m/water_level_rv02_m
    # for genuinely per-river conditions (e.g. "Narayani rising, Rapti
    # constant" -- architecturally impossible to represent with a single
    # water_level_m, since every grid shared the exact same number).
    water_level_m: float | None = Query(default=None, ge=0),
    water_level_rv01_m: float | None = Query(
        default=None, ge=0, description="Narayani/Devghat gauge reading, applied only to grids near that river."
    ),
    water_level_rv02_m: float | None = Query(
        default=None, ge=0, description="East Rapti/Rajaiya gauge reading, applied only to grids near that river."
    ),
    # river_station_id is still accepted so existing callers don't 422, but
    # no longer forces every grid onto one station -- see
    # _resolve_grid_conditions()'s docstring for why. warning_level_m/
    # danger_level_m are unchanged (still applied globally when supplied,
    # same as before this fix -- BIPAD only ever returns one station's
    # thresholds per call today, so there's no per-station equivalent to
    # build here yet without a frontend/BIPAD-client change too).
    river_station_id: str | None = Query(default=None),
    warning_level_m: float | None = Query(default=None, ge=0),
    danger_level_m: float | None = Query(default=None, ge=0),
) -> dict[str, Any]:
    if water_level_m is None and water_level_rv01_m is None and water_level_rv02_m is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing water level",
                "message": (
                    "Provide water_level_m (legacy, applied uniformly to every grid) and/or "
                    "water_level_rv01_m / water_level_rv02_m (per-station, preferred)."
                ),
            },
        )
    return {
        "rainfall_24h_mm": rainfall_24h_mm,
        "water_level_m": water_level_m,
        "water_level_by_station": {"rv01": water_level_rv01_m, "rv02": water_level_rv02_m},
        "river_station_id": river_station_id,
        "warning_level_m": warning_level_m,
        "danger_level_m": danger_level_m,
    }


@router.get("/risk-map")
def get_risk_map(
    model: str = Query(default=BEST_MODEL_ID),
    conditions: dict[str, Any] = Depends(condition_params),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Score every grid with the chosen model under the given basin-wide
    conditions and return it as a GeoJSON FeatureCollection.

    Grids with missing required static fields are skipped and listed under
    `skipped_grids` rather than failing the whole map.

    Also logs High/Danger grids to RiskResultLog, debounced (see
    _log_risk_result_if_due above and GET /api/history/risk-results) --
    doubles as this project's alert history, see app/db/models.py.
    """
    enriched_features, skipped = score_all_grids(model, conditions)
    _log_risk_result_if_due(db, model, conditions, enriched_features)
    collection = build_feature_collection(enriched_features)
    collection["skipped_grids"] = skipped
    # BREAKING CHANGE (Fix 3, RIVER_STATION_RESOLUTION_BACKEND_PROMPT.md):
    # this used to be a single top-level boolean, which only ever made
    # sense because every grid shared one global condition/station. Now
    # that grids can resolve to different stations/readings, "was the
    # fallback used" isn't a single yes/no across the whole map -- some
    # grids may use real thresholds while others fall back. Replaced with a
    # count. Frontend must be updated to read this instead of the old
    # boolean field, which no longer exists in the response.
    collection["grids_using_river_fallback"] = sum(
        1 for f in enriched_features if f["properties"]["river_level_score_used_fallback"]
    )
    return collection


@router.get("/risk-map/affected-area")
def get_affected_area(
    model: str = Query(default=BEST_MODEL_ID),
    conditions: dict[str, Any] = Depends(condition_params),
) -> dict[str, Any]:
    """Dissolve all High/Danger grids (for the chosen model + conditions)
    into a single affected-area polygon."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "geopandas not installed",
                "message": "Install geopandas (see requirements.txt) to use this endpoint.",
            },
        ) from exc

    enriched_features, _skipped = score_all_grids(model, conditions)

    affected_features = [
        f for f in enriched_features if f["properties"]["predicted_risk_class"] in ("High", "Danger")
    ]

    if not affected_features:
        return {"type": "Feature", "properties": {"model_id": model, "grid_count": 0}, "geometry": None}

    # get_geojson_data() already reprojects to EPSG:4326 (see app/utils.py),
    # so these features are lon/lat, not the source file's UTM meters.
    gdf = gpd.GeoDataFrame.from_features(
        build_feature_collection(affected_features)["features"], crs="EPSG:4326"
    )
    gdf["dissolve_key"] = 1
    dissolved = gdf.dissolve(by="dissolve_key")

    grid_count = len(affected_features)
    geometry = dissolved.geometry.iloc[0]

    return {
        "type": "Feature",
        "properties": {"model_id": model, "grid_count": grid_count},
        "geometry": geometry.__geo_interface__,
    }
