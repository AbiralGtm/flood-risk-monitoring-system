"""Database engine/session setup.

Defaults to a local SQLite file (zero setup, matches the proposal's
"economically feasible... open-source and freely available" framing for a
prototype) so this works out of the box with no external service. Proposal
Table 1.1 specifies "MySQL / Spatial SQL Database" for the eventual system --
set the DATABASE_URL environment variable to switch, e.g.:

    DATABASE_URL=mysql+pymysql://user:password@host/flood_risk

SQLAlchemy's engine/session API is identical either way; only the driver
(pymysql, psycopg2, etc.) needs installing for a non-SQLite target -- see
requirements.txt.

This intentionally does NOT persist grid geometry/features (GRID_FEATURES in
the proposal's ER diagram) -- those still come from the static GeoJSON file
(app/utils.get_geojson_data). What's stored here is *history*: rainfall/
river-level readings, forecast runs, per-grid prediction requests, and
alert-worthy risk-map results -- the tables the status report flagged as
missing ("no historical record of past predictions").

Schema changes go through Alembic (migrations/, alembic.ini) as of 2026-08 --
see init_db()'s docstring below for how that interacts with the dev-startup
create_all() call.
"""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import BASE_DIR

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR / 'flood_risk.db'}")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a session, always closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables that don't exist yet. Safe to call on every startup --
    only adds missing tables, never alters or drops existing ones, so it
    can't conflict with Alembic-applied changes.

    Kept as a zero-friction dev/demo fallback (fresh SQLite file, no
    migration step required to just run the app) now that Alembic exists --
    for any change to a table that already has real data in it, use
    `alembic revision --autogenerate` + `alembic upgrade head` instead of
    relying on this, since create_all() cannot alter existing tables.
    """
    from app.db import models  # noqa: F401  (import needed so Base knows about the tables)

    Base.metadata.create_all(bind=engine)
