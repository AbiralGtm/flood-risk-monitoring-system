import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import PredictionLog, RainfallRecord, RiverLevelRecord
from app.db.session import get_db
from app.early_warning import get_early_warning
from app.ml.explain import compute_contributing_factors
from app.ml.features import build_feature_row, station_id_aliases
from app.ml.model_loader import BEST_MODEL_ID, get_model, list_model_ids
from app.utils import get_feature_by_grid_id

router = APIRouter()


class DynamicConditions(BaseModel):
    """Live or forecast conditions applied to a grid at prediction time.

    rainfall_1h_mm/3h_mm/6h_mm/12h_mm were removed 2026-08: the model no
    longer uses them (see app/ml/features.py module docstring), and if the
    frontend still sends them they're simply ignored (pydantic ignores
    unrecognized extra fields by default) rather than causing a validation
    error, so this is not a breaking change for existing callers.
    """

    rainfall_24h_mm: float = Field(ge=0)
    water_level_m: float = Field(ge=0)
    river_station_id: str | None = None
    warning_level_m: float | None = Field(default=None, ge=0)
    danger_level_m: float | None = Field(default=None, ge=0)


class PredictRequest(BaseModel):
    grid_id: str
    conditions: DynamicConditions


def _feature_row_for_grid(grid_id: str, conditions: DynamicConditions) -> dict[str, float]:
    feature = get_feature_by_grid_id(grid_id)
    properties = feature.get("properties", {})
    try:
        return build_feature_row(properties, conditions.model_dump())
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail={"error": "Not implemented", "message": str(exc)}) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "Feature build failed", "message": str(exc)},
        ) from exc


def _diagnostics(feature_row: dict[str, Any]) -> dict[str, Any]:
    """Surfaces conditions that silently affect the prediction but are easy to get
    wrong from the caller's side (e.g. omitting river_station_id). Two requests
    for the same grid that differ only in whether these line up can otherwise
    look like a prediction bug when they're actually just different requests.
    """
    return {
        "river_level_score": feature_row["river_level_score"],
        "river_level_score_used_fallback": feature_row["_river_level_score_used_fallback"],
        "river_station_id": feature_row["_river_station_id"],
    }


def _log_prediction(
    db: Session,
    grid_id: str,
    result: dict[str, Any],
    feature_row: dict[str, Any],
    water_level_m: float,
) -> None:
    """Best-effort history write (see app/db/models.py PredictionLog docstring
    for scope). A DB hiccup must never fail a prediction response.

    water_level_m is passed separately (not read off feature_row) since
    build_feature_row() only uses it to derive river_level_score and doesn't
    include the raw value in the returned dict (see app/ml/features.py).
    """
    try:
        db.add(
            PredictionLog(
                grid_id=grid_id,
                model_id=result["model_id"],
                predicted_risk_class=result["predicted_risk_class"],
                risk_probability=result["risk_probability"],
                river_level_score=feature_row["river_level_score"],
                river_level_score_used_fallback=feature_row["_river_level_score_used_fallback"],
                rainfall_24h_mm=feature_row["rainfall_24h_mm"],
                water_level_m=water_level_m,
            )
        )
        db.commit()
    except Exception:  # noqa: BLE001 - logging is best-effort, never fail the request over it
        db.rollback()


def _early_warning_for_grid(db: Session, feature_row: dict[str, Any]) -> dict[str, Any]:
    """Builds get_early_warning()'s two inputs from logged history and calls
    it (see app/early_warning.py). Independent of the trained model -- never
    touches risk_class/probabilities, purely additive.

    Best-effort: any DB problem here must never fail the prediction response
    itself (same convention as _log_prediction below), so this degrades to
    "no warning" rather than raising -- matching get_early_warning()'s own
    documented behavior of returning early_warning=False on empty inputs
    rather than crashing.
    """
    try:
        station_id = feature_row.get("_river_station_id")
        aliases = station_id_aliases(station_id)

        river_readings: list[tuple[datetime, float]] = []
        if aliases:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            rows = (
                db.query(RiverLevelRecord)
                .filter(RiverLevelRecord.station_id.in_(aliases), RiverLevelRecord.created_at >= cutoff)
                .all()
            )
            river_readings = [(row.created_at, row.water_level_m) for row in rows]

        # Rainfall is basin-wide in this system, not logged per station (see
        # app/services/bipad_client.py -- BIPAD's 3 active Bharatpur-area rain
        # gauges are averaged into one rainfall_24h_mm before it's ever
        # returned or logged; RainfallRecord only stores that blended figure
        # plus which station names contributed, not each one's own value).
        # There's no genuine per-station rainfall history to draw from, so
        # this maps every contributing station name from the most recent
        # logged reading to that SAME blended value -- an honest
        # approximation of "use all 3 basin-wide gauges" given what's
        # actually stored, not independently-verified per-station data. If
        # true per-station rainfall history is ever needed, bipad_client.py's
        # fetch would need to stop collapsing per-station values before
        # averaging.
        catchment_rainfall_mm: dict[str, float] = {}
        latest_rainfall = (
            db.query(RainfallRecord).order_by(RainfallRecord.created_at.desc()).first()
        )
        if latest_rainfall is not None and latest_rainfall.rain_stations_used:
            station_names = [s.strip() for s in latest_rainfall.rain_stations_used.split(",") if s.strip()]
            catchment_rainfall_mm = {name: latest_rainfall.rainfall_24h_mm for name in station_names}

        return get_early_warning(river_readings=river_readings, catchment_rainfall_mm=catchment_rainfall_mm)
    except Exception:  # noqa: BLE001 - best-effort, never fail the prediction response over it
        return {"early_warning": False, "early_warning_reasons": []}


@router.post("/predict")
def predict(request: PredictRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Predict flood risk for one grid using the best model (XGBoost).

    Logs the request to PredictionLog on success (see app/routes/history.py's
    GET /api/history/predictions) -- /predict/compare does not log, to avoid
    3x-counting the same grid/conditions (see app/db/models.py docstring).

    Does NOT log alerts -- alerts now come from /api/risk-map (map-wide,
    debounced), see app/routes/risk_map.py and app/db/models.py's
    RiskResultLog docstring for why (DATABASE_FULL_BACKEND_PROMPT.md
    Decision 1: one table for both RISK_RESULTS and ALERTS, not a separate
    one built off single-grid predictions).
    """
    feature_row = _feature_row_for_grid(request.grid_id, request.conditions)
    bundle = get_model(BEST_MODEL_ID)
    result = bundle.predict(feature_row)
    _log_prediction(db, request.grid_id, result, feature_row, request.conditions.water_level_m)
    # Independent early-warning layer (app/early_warning.py) -- computed
    # after the model's own prediction, never alters risk_class/probabilities,
    # purely additive fields on the response. See _early_warning_for_grid().
    early_warning = _early_warning_for_grid(db, feature_row)
    return {
        "grid_id": request.grid_id,
        **result,
        "early_warning": early_warning["early_warning"],
        "early_warning_reasons": early_warning["early_warning_reasons"],
        "diagnostics": _diagnostics(feature_row),
        "contributing_factors": compute_contributing_factors(feature_row),
    }


@router.post("/predict/compare")
def predict_compare(request: PredictRequest) -> dict[str, Any]:
    """Predict flood risk for one grid using all available models, for comparison.

    If an individual model fails -- e.g. a library version mismatch against
    what it was pickled with, or a corrupted/incomplete .joblib file on disk --
    that model's entry reports the error instead of failing the whole
    comparison. Loading (get_model) happens inside the loop, per model, so a
    model that fails to even load doesn't take down the other two.

    contributing_factors describes the input feature values, not any one
    model's output -- the same build_feature_row() result feeds all three
    models, so it's computed once here at the top level (alongside
    best_model/diagnostics), not duplicated inside each per-model entry.
    """
    feature_row = _feature_row_for_grid(request.grid_id, request.conditions)

    results: dict[str, Any] = {}
    for model_id in list_model_ids():
        try:
            bundle = get_model(model_id)
            results[model_id] = bundle.predict(feature_row)
        except Exception as exc:  # noqa: BLE001 - surface per-model failure, don't fail the whole comparison
            results[model_id] = {"model_id": model_id, "error": str(exc)}

    return {
        "grid_id": request.grid_id,
        "best_model": BEST_MODEL_ID,
        "predictions": results,
        "diagnostics": _diagnostics(feature_row),
        "contributing_factors": compute_contributing_factors(feature_row),
    }


@router.get("/models")
def list_models() -> dict[str, Any]:
    """List available models with metadata and stored validation metrics.

    A model that fails to load is still listed, with an `error` field, rather
    than taking down this endpoint for the other models.
    """
    models_meta = []
    for model_id in list_model_ids():
        try:
            bundle = get_model(model_id)
        except Exception as exc:  # noqa: BLE001 - surface per-model failure, don't fail the whole list
            models_meta.append({"model_id": model_id, "error": str(exc)})
            continue
        models_meta.append(
            {
                "model_id": bundle.model_id,
                "display_name": bundle.display_name,
                "algorithm": bundle.algorithm,
                "is_best": bundle.model_id == BEST_MODEL_ID,
                "classes": bundle.classes,
                "metrics": bundle.metrics,
            }
        )
    return {"best_model": BEST_MODEL_ID, "models": models_meta}


@router.get("/models/evaluation-info")
def get_evaluation_info() -> dict[str, Any]:
    """Metadata about when/how the metrics in GET /api/models were computed --
    NOT a live recompute (that reads a 1.6GB dataset and takes real wall-clock
    time, not something to run inline in a request). Reads app/ml/eval_metadata.json,
    written by `python -m app.ml.evaluate` (see app/ml/evaluate.py). Use this to
    catch "these numbers are stale" -- e.g. after a model fine-tuning pass that
    didn't re-run the eval script -- rather than assuming MODEL_METRICS is
    always current.
    """
    eval_metadata_path = Path(__file__).resolve().parent.parent / "ml" / "eval_metadata.json"
    if not eval_metadata_path.exists():
        return {
            "available": False,
            "message": (
                "No recorded evaluation run found. Run `python -m app.ml.evaluate` "
                "to generate app/ml/eval_metadata.json."
            ),
        }
    try:
        metadata = json.loads(eval_metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "Could not read evaluation metadata", "message": str(exc)},
        ) from exc
    return {"available": True, **metadata}


@router.get("/models/gis-output-test")
def get_gis_output_test() -> dict[str, Any]:
    """Results of the GIS output test -- proposal Section 3.13.2: predicted
    High/Danger zones vs. historical flood extent, evaluated via Intersection
    over Union. NOT a live recompute (scores ~1.7M historical rows against the
    live model, takes ~2 minutes). Reads app/ml/gis_output_test_results.json,
    written by `python -m app.ml.gis_output_test` (see that module's docstring
    for the reconstructed-vs-historical_evidence_override methodology caveat --
    read `results.historical_evidence_override_only` for the methodologically
    clean number, not just `results.combined_all_historical_rows`).
    """
    results_path = Path(__file__).resolve().parent.parent / "ml" / "gis_output_test_results.json"
    if not results_path.exists():
        return {
            "available": False,
            "message": (
                "No recorded GIS output test run found. Run `python -m app.ml.gis_output_test` "
                "to generate app/ml/gis_output_test_results.json."
            ),
        }
    try:
        results = json.loads(results_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "Could not read GIS output test results", "message": str(exc)},
        ) from exc
    return {"available": True, **results}
