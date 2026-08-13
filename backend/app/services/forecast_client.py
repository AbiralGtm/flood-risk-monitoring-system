"""Fetches forward-looking rainfall forecasts from Open-Meteo's free public API.

Open-Meteo (https://open-meteo.com) requires no API key and is free for
non-commercial use (10,000 calls/day; CC BY 4.0 attribution required -- see
https://open-meteo.com/en/terms). It blends several real NWP models
(including ECMWF and GFS) and auto-selects the best available for a given
location. Verified live 2026-08-04 by fetching hourly + daily precipitation
for the study area's centroid and confirming genuine multi-day forward-
looking values (not a nowcast/estimate of rain that already fell, unlike
GPM IMERG -- see FORECAST_MODE_BACKEND_PROMPT.md research).

    GET https://api.open-meteo.com/v1/forecast
        ?latitude=27.6188&longitude=84.3329
        &hourly=precipitation&forecast_days=4&timezone=Asia/Kathmandu
    -> {"hourly": {"time": [...], "precipitation": [...]}}

This project's model has exactly one rainfall input, rainfall_24h_mm (see
app/ml/features.py module docstring) -- there is no per-horizon-aware model.
Per FORECAST_MODE_BACKEND_PROMPT.md Option A/C: until/unless that changes,
a forecast horizon request should return the cumulative rainfall expected
over that horizon window (0 to Nh from now), which the frontend then uses
as its "best estimate for the horizon" rainfall_24h_mm input to the
existing /api/predict and /api/risk-map endpoints -- not a distinct model
feature. That's an honest v1 design, not a stopgap.

Open-Meteo has no river-level forecast product; forecast mode's water_level_m
is intentionally NOT sourced here -- callers should keep using the current/
live river reading or a manually entered value for that part of a forecast
scenario.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 10

# Centroid of the canonical 2,434-grid study-area dataset
# (data/static_grid_dataset_with_stations.geojson), computed 2026-08-04 --
# a single representative point for a basin-wide forecast rainfall figure,
# consistent with how /api/risk-map already applies one set of dynamic
# conditions uniformly across the whole map per request.
STUDY_AREA_LATITUDE = 27.6188
STUDY_AREA_LONGITUDE = 84.3329

HORIZON_HOURS: dict[str, int] = {"24h": 24, "48h": 48, "72h": 72}

# Forecasts don't change nearly as fast as live gauge readings -- Open-Meteo's
# underlying NWP model runs update every few hours, so a longer cache than
# bipad_client's 5 minutes avoids burning free-tier calls for no benefit.
_CACHE_TTL_SECONDS = 1800
_cache: dict[str, dict[str, Any]] = {}


class ForecastFetchError(RuntimeError):
    """Raised when Open-Meteo's API is unreachable or returns unexpected data."""


def _get_hourly_precipitation() -> tuple[list[str], list[float]]:
    params = {
        "latitude": STUDY_AREA_LATITUDE,
        "longitude": STUDY_AREA_LONGITUDE,
        "hourly": "precipitation",
        "forecast_days": 4,  # need 72h of *future* hours; 4 days covers that with margin
        "timezone": "Asia/Kathmandu",
    }
    try:
        response = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ForecastFetchError(f"Request to Open-Meteo failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ForecastFetchError("Non-JSON response from Open-Meteo") from exc

    hourly = payload.get("hourly") or {}
    times = hourly.get("time")
    values = hourly.get("precipitation")
    if not times or not values or len(times) != len(values):
        raise ForecastFetchError(f"Unexpected Open-Meteo response shape: {payload}")

    return times, values


def get_forecast_conditions(horizon: str) -> dict[str, Any]:
    """Cumulative rainfall forecast for the next `horizon` (one of "24h",
    "48h", "72h") from now, sourced from Open-Meteo.

    Cached per-horizon for _CACHE_TTL_SECONDS. Raises ForecastFetchError if
    Open-Meteo can't be reached -- callers must surface that as a clear
    error, never silently substitute a guessed rainfall value (same rule
    app/services/bipad_client.py follows for live conditions).
    """
    if horizon not in HORIZON_HOURS:
        raise ValueError(f"Unknown horizon '{horizon}'. Expected one of {list(HORIZON_HOURS)}.")

    now = time.monotonic()
    cached = _cache.get(horizon)
    if cached is not None and (now - cached["fetched_at"]) < _CACHE_TTL_SECONDS:
        return cached["value"]

    times, values = _get_hourly_precipitation()

    # Open-Meteo's hourly series starts at the current local hour (verified
    # live 2026-08-04: first entry was "today"'s midnight-aligned hour, with
    # the series covering past-today's-start through forecast_days ahead --
    # so we anchor on "now" explicitly rather than assuming index 0 is "now").
    now_local = datetime.now(NEPAL_TZ).replace(minute=0, second=0, microsecond=0)
    now_str = now_local.strftime("%Y-%m-%dT%H:00")

    try:
        start_idx = times.index(now_str)
    except ValueError:
        # Current hour not in the returned series (e.g. right at a day boundary
        # during Open-Meteo's own data refresh) -- fail loud rather than guess
        # an offset that could silently sum the wrong window.
        raise ForecastFetchError(
            f"Could not locate current hour '{now_str}' in Open-Meteo's hourly series "
            f"(series spans {times[0]} to {times[-1]})"
        )

    hours_needed = HORIZON_HOURS[horizon]
    window = values[start_idx : start_idx + hours_needed]
    if len(window) < hours_needed:
        raise ForecastFetchError(
            f"Open-Meteo returned only {len(window)} forecast hours from now, "
            f"need {hours_needed} for horizon '{horizon}'"
        )

    rainfall_mm = round(sum(v for v in window if v is not None), 2)

    result = {
        "horizon": horizon,
        "rainfall_24h_mm": rainfall_mm,  # the model's actual feature name -- see module docstring
        "window_start": times[start_idx],
        "window_end": times[start_idx + hours_needed - 1],
        "source": "open-meteo",
        "fetched_at": datetime.now(NEPAL_TZ).isoformat(),
    }

    _cache[horizon] = {"value": result, "fetched_at": now}
    return result
