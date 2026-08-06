"""ORM models for the history tables -- a scoped subset of the proposal's
ER diagram (Figure 3.3).

Deliberately NOT included yet, and why:
  - GRID_FEATURES: still served from the static GeoJSON file, not the DB.
    Migrating that is a separate decision (would need a real spatial column
    type -- GEOMETRY on MySQL/PostGIS -- that SQLite can't represent
    natively), not required for "show trend history."
  - RESULT_AFFECTED_AREA / AFFECTED_AREAS: /api/risk-map/affected-area's
    dissolved polygon is cheap to recompute on demand; not persisting the
    polygon itself (see DATABASE_FULL_BACKEND_PROMPT.md Decision 2 -- decided
    against, nothing currently needs "what did the affected area look like
    at time T").
  - USERS: no auth exists yet.

PredictionLog covers /api/predict (one grid at a time, naturally bounded).

RiskResultLog (added 2026-08, replacing an earlier single-grid-only
AlertRecord -- see DATABASE_FULL_BACKEND_PROMPT.md Decision 1, "one table
closes both RISK_RESULTS and ALERTS") covers /api/risk-map runs, but ONLY
rows for grids at High/Danger, and only for runs that pass the debounce
check in app/routes/risk_map.py -- logging all 2,434 grids on every
interactive map drag would explode the table, so this intentionally does
NOT give a complete per-grid history of every risk-map call, only a
alert-worthy-grids-over-time view. `run_id` groups rows written by the same
risk-map call, so one map interaction's alerts can be queried together.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RainfallRecord(Base):
    """One rainfall reading/observation, from BIPAD (live) or manual entry.
    Maps to RAINFALL_RECORDS in the proposal's ER diagram."""

    __tablename__ = "rainfall_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))  # "bipad" | "manual"
    rainfall_24h_mm: Mapped[float] = mapped_column(Float)
    rain_stations_used: Mapped[str | None] = mapped_column(String(255), nullable=True)
    recorded_at: Mapped[str | None] = mapped_column(String(64), nullable=True)  # source's own timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RiverLevelRecord(Base):
    """One river-gauge reading, from BIPAD (live) or manual entry.
    Maps to RIVER_LEVEL_RECORDS in the proposal's ER diagram."""

    __tablename__ = "river_level_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))  # "bipad" | "manual"
    station_id: Mapped[str] = mapped_column(String(64))
    water_level_m: Mapped[float] = mapped_column(Float)
    warning_level_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    danger_level_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ForecastRecord(Base):
    """One forecast fetch (horizon + resulting rainfall figure).
    Maps to FORECAST_RECORDS in the proposal's ER diagram."""

    __tablename__ = "forecast_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    horizon: Mapped[str] = mapped_column(String(8))  # "24h" | "48h" | "72h"
    rainfall_24h_mm: Mapped[float] = mapped_column(Float)
    window_start: Mapped[str | None] = mapped_column(String(32), nullable=True)
    window_end: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="open-meteo")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PredictionLog(Base):
    """One /api/predict call for one grid. Maps to RISK_RESULTS in the
    proposal's ER diagram, scoped to single-grid requests (see module
    docstring for why /api/risk-map's bulk scoring isn't logged here)."""

    __tablename__ = "prediction_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    grid_id: Mapped[str] = mapped_column(String(32), index=True)
    model_id: Mapped[str] = mapped_column(String(32))
    predicted_risk_class: Mapped[str] = mapped_column(String(16))
    risk_probability: Mapped[float] = mapped_column(Float)
    river_level_score: Mapped[float] = mapped_column(Float)
    river_level_score_used_fallback: Mapped[bool] = mapped_column(Boolean)
    rainfall_24h_mm: Mapped[float] = mapped_column(Float)
    water_level_m: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class RiskResultLog(Base):
    """One grid's result from a /api/risk-map run, logged only when the grid
    is High/Danger -- doubles as the ALERTS record (see
    DATABASE_FULL_BACKEND_PROMPT.md Decision 1; this replaces the earlier
    single-grid-only AlertRecord). Maps to RISK_RESULTS + ALERTS in the
    proposal's ER diagram, for the map-wide (not single-grid) case.

    Not every /api/risk-map call logs -- see the debounce check in
    app/routes/risk_map.py, which skips logging if the same conditions
    fingerprint (rainfall+water_level+model, rounded) was already logged in
    the last 60s, so interactive slider-dragging doesn't spam this table.
    """

    __tablename__ = "risk_result_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    grid_id: Mapped[str] = mapped_column(String(32), index=True)
    model_id: Mapped[str] = mapped_column(String(32))
    predicted_risk_class: Mapped[str] = mapped_column(String(16))  # "High" | "Danger" only
    rainfall_24h_mm: Mapped[float] = mapped_column(Float)
    water_level_m: Mapped[float] = mapped_column(Float)
    run_id: Mapped[str] = mapped_column(String(36), index=True)  # groups rows from the same risk-map call
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
