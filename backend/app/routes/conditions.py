from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.models import ForecastRecord, RainfallRecord, RiverLevelRecord
from app.db.session import get_db
from app.services.bipad_client import BipadFetchError, get_live_conditions
from app.services.forecast_client import ForecastFetchError, get_forecast_conditions

router = APIRouter()


def _log_live_conditions(db: Session, conditions: dict[str, Any], river_station_id: str) -> None:
    """Best-effort history write. A DB hiccup must never take down a live
    conditions response -- callers already depend on this endpoint to
    pre-fill the form, so the read path stays authoritative even if logging
    fails."""
    try:
        db.add(
            RainfallRecord(
                source="bipad",
                rainfall_24h_mm=conditions["rainfall_24h_mm"],
                rain_stations_used=", ".join(conditions.get("rain_stations_used") or []) or None,
                recorded_at=conditions.get("fetched_at"),
            )
        )
        db.add(
            RiverLevelRecord(
                source="bipad",
                station_id=river_station_id,
                water_level_m=conditions["water_level_m"],
                warning_level_m=conditions.get("warning_level_m"),
                danger_level_m=conditions.get("danger_level_m"),
                recorded_at=conditions.get("fetched_at"),
            )
        )
        db.commit()
    except Exception:  # noqa: BLE001 - logging is best-effort, never fail the request over it
        db.rollback()


def _log_forecast(db: Session, forecast: dict[str, Any]) -> None:
    try:
        db.add(
            ForecastRecord(
                horizon=forecast["horizon"],
                rainfall_24h_mm=forecast["rainfall_24h_mm"],
                window_start=forecast.get("window_start"),
                window_end=forecast.get("window_end"),
                source=forecast.get("source", "open-meteo"),
            )
        )
        db.commit()
    except Exception:  # noqa: BLE001 - logging is best-effort, never fail the request over it
        db.rollback()


@router.get("/conditions/live")
def get_conditions_live(
    river_station_id: str = Query(default="devghat", pattern="^(devghat|rajaiya)$"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Live rainfall (averaged across active Bharatpur-area BIPAD gauges) and
    river level (from the given station) for pre-filling the conditions
    form / driving /api/risk-map automatically.

    Returns 502 if BIPAD can't be reached -- callers must not treat that as
    "conditions are calm," only as "live data unavailable right now."

    On success, also logs the reading to the history tables (see
    app/routes/history.py) -- logging failures are swallowed, never surfaced
    as a 502 for what was otherwise a successful fetch.
    """
    try:
        conditions = get_live_conditions(primary_river_station=river_station_id)
    except BipadFetchError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Live conditions unavailable",
                "message": str(exc),
            },
        ) from exc

    _log_live_conditions(db, conditions, river_station_id)
    return conditions


@router.get("/conditions/forecast")
def get_conditions_forecast(
    horizon: str = Query(..., pattern="^(24h|48h|72h)$"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Forecast rainfall (Open-Meteo, cumulative over the next `horizon`) for
    pre-filling Forecast Mode's conditions form.

    There is no river-level forecast source wired in -- Open-Meteo doesn't
    provide one. Callers should keep using the current/live river reading
    (GET /api/conditions/live) or a manually entered value for water_level_m
    as part of a forecast scenario; this endpoint only covers rainfall.

    The response's `rainfall_24h_mm` field is named to match the model's
    actual feature (see app/ml/features.py) -- it's the cumulative rainfall
    forecast for the *entire* requested horizon window, not literally "the
    next 24 hours" when horizon is 48h/72h. Pass it straight through as the
    rainfall_24h_mm value in a /api/predict or /api/risk-map conditions
    payload representing "my best estimate for this horizon."

    Returns 502 if Open-Meteo can't be reached -- same rule as /conditions/live:
    never silently substitute a guessed rainfall value.
    """
    try:
        forecast = get_forecast_conditions(horizon)
    except ForecastFetchError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Forecast conditions unavailable",
                "message": str(exc),
            },
        ) from exc

    _log_forecast(db, forecast)
    return forecast
