"""Read endpoints over the history tables (app/db/models.py).

These serve the proposal's "historical record of past predictions / past
conditions" requirement (status report's previously-flagged database gap).
Nothing here computes anything new -- it's a thin read layer over rows
written by other routes (see app/routes/conditions.py and
app/routes/predict.py for the write side).

All endpoints are read-only and return most-recent-first, capped by a
`limit` query param (default 50, max 500) so a forgotten filter can't pull
an unbounded table.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ForecastRecord, PredictionLog, RainfallRecord, RiskResultLog, RiverLevelRecord
from app.db.session import get_db

router = APIRouter()

_MAX_LIMIT = 500


def _rainfall_to_dict(row: RainfallRecord) -> dict[str, Any]:
    return {
        "id": row.id,
        "source": row.source,
        "rainfall_24h_mm": row.rainfall_24h_mm,
        "rain_stations_used": row.rain_stations_used,
        "recorded_at": row.recorded_at,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _river_level_to_dict(row: RiverLevelRecord) -> dict[str, Any]:
    return {
        "id": row.id,
        "source": row.source,
        "station_id": row.station_id,
        "water_level_m": row.water_level_m,
        "warning_level_m": row.warning_level_m,
        "danger_level_m": row.danger_level_m,
        "recorded_at": row.recorded_at,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _forecast_to_dict(row: ForecastRecord) -> dict[str, Any]:
    return {
        "id": row.id,
        "horizon": row.horizon,
        "rainfall_24h_mm": row.rainfall_24h_mm,
        "window_start": row.window_start,
        "window_end": row.window_end,
        "source": row.source,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _risk_result_to_dict(row: RiskResultLog) -> dict[str, Any]:
    return {
        "id": row.id,
        "grid_id": row.grid_id,
        "model_id": row.model_id,
        "predicted_risk_class": row.predicted_risk_class,
        "rainfall_24h_mm": row.rainfall_24h_mm,
        "water_level_m": row.water_level_m,
        "run_id": row.run_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _prediction_to_dict(row: PredictionLog) -> dict[str, Any]:
    return {
        "id": row.id,
        "grid_id": row.grid_id,
        "model_id": row.model_id,
        "predicted_risk_class": row.predicted_risk_class,
        "risk_probability": row.risk_probability,
        "river_level_score": row.river_level_score,
        "river_level_score_used_fallback": row.river_level_score_used_fallback,
        "rainfall_24h_mm": row.rainfall_24h_mm,
        "water_level_m": row.water_level_m,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/history/rainfall")
def get_rainfall_history(
    limit: int = Query(default=50, ge=1, le=_MAX_LIMIT),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Most recent rainfall readings written by GET /api/conditions/live."""
    rows = db.execute(
        select(RainfallRecord).order_by(RainfallRecord.created_at.desc()).limit(limit)
    ).scalars().all()
    return {"count": len(rows), "records": [_rainfall_to_dict(r) for r in rows]}


@router.get("/history/river-level")
def get_river_level_history(
    station_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=_MAX_LIMIT),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Most recent river-level readings written by GET /api/conditions/live.
    Optionally filtered to a single station_id (e.g. "devghat")."""
    stmt = select(RiverLevelRecord)
    if station_id:
        stmt = stmt.where(RiverLevelRecord.station_id == station_id)
    stmt = stmt.order_by(RiverLevelRecord.created_at.desc()).limit(limit)
    rows = db.execute(stmt).scalars().all()
    return {"count": len(rows), "records": [_river_level_to_dict(r) for r in rows]}


@router.get("/history/forecasts")
def get_forecast_history(
    horizon: str | None = Query(default=None, pattern="^(24h|48h|72h)$"),
    limit: int = Query(default=50, ge=1, le=_MAX_LIMIT),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Most recent forecast fetches written by GET /api/conditions/forecast.
    Optionally filtered to a single horizon."""
    stmt = select(ForecastRecord)
    if horizon:
        stmt = stmt.where(ForecastRecord.horizon == horizon)
    stmt = stmt.order_by(ForecastRecord.created_at.desc()).limit(limit)
    rows = db.execute(stmt).scalars().all()
    return {"count": len(rows), "records": [_forecast_to_dict(r) for r in rows]}


@router.get("/history/risk-results")
def get_risk_result_history(
    grid_id: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=_MAX_LIMIT),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Most recent High/Danger grids logged from GET /api/risk-map runs (see
    app/db/models.py RiskResultLog docstring) -- doubles as this project's
    alert history. Optionally filtered to a single grid_id and/or run_id
    (run_id groups every row written by the same risk-map call). Debounced
    at write time (see app/routes/risk_map.py), so this is "alert-worthy
    grids over time," not a complete per-request log of every risk-map call."""
    stmt = select(RiskResultLog)
    if grid_id:
        stmt = stmt.where(RiskResultLog.grid_id == grid_id)
    if run_id:
        stmt = stmt.where(RiskResultLog.run_id == run_id)
    stmt = stmt.order_by(RiskResultLog.created_at.desc()).limit(limit)
    rows = db.execute(stmt).scalars().all()
    return {"count": len(rows), "records": [_risk_result_to_dict(r) for r in rows]}


@router.get("/history/predictions")
def get_prediction_history(
    grid_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=_MAX_LIMIT),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Most recent single-grid predictions written by POST /api/predict.
    Optionally filtered to a single grid_id. Does NOT include /api/predict/compare
    or /api/risk-map runs -- see app/db/models.py's PredictionLog docstring
    for why those aren't persisted here."""
    stmt = select(PredictionLog)
    if grid_id:
        stmt = stmt.where(PredictionLog.grid_id == grid_id)
    stmt = stmt.order_by(PredictionLog.created_at.desc()).limit(limit)
    rows = db.execute(stmt).scalars().all()
    return {"count": len(rows), "records": [_prediction_to_dict(r) for r in rows]}
