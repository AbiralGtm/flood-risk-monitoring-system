"""Fetches live rainfall + river-level readings from BIPAD Portal's public API.

BIPAD (bipadportal.gov.np) has no published API docs, but its own realtime
dashboard (https://bipadportal.gov.np/realtime/) calls a plain, unauthenticated
Django REST Framework API under /api/v1/. Confirmed live and working
2026-07-27 by inspecting that dashboard's network requests:

    GET https://bipadportal.gov.np/api/v1/river/
        ?search=<station name>&ordering=-water_level_on&limit=1&format=json
        &fields=id,title,water_level,danger_level,warning_level,water_level_on,status
    -> {"results": [{"title": "Narayani at Devghat", "waterLevel": 4.72,
                      "dangerLevel": 9.0, "warningLevel": 7.3,
                      "waterLevelOn": "2026-07-27T19:20:00+05:45", ...}]}

    GET https://bipadportal.gov.np/api/v1/rain/
        ?station=<station id>&ordering=-measured_on&limit=1&format=json
    -> {"results": [{"title": "Rampur_AWS", "measuredOn": "...",
                      "averages": [{"interval": 1, "value": 0.0}, ...
                                    {"interval": 24, "value": 12.4}], ...}]}

Both endpoints use DRF-style ordering (snake_case field name: water_level_on,
measured_on) -- the camelCase equivalents shown in the JSON response do NOT
work as ordering params, only as output field names. `search=` does a
full-text match and can return stale historical rows mixed with current
ones, which is why every call here also orders by the real timestamp field
and takes the single latest one, rather than trusting search result order.

Two river gauges matter for this project (see RIVER_STATION_THRESHOLDS in
app/ml/features.py, sourced from this same API):
    devghat  -> "Narayani at Devghat"   (warning 7.3m / danger 9.0m)
    rajaiya  -> "East Rapti at Rajaiya" (warning 3.3m / danger 3.7m)

Rainfall: Bharatpur Metropolitan City's municipality id in BIPAD is 35001.
As of 2026-07-27, three rain gauges inside that municipality are actively
reporting hourly (station ids below); a fourth, "Bharatpur Airport" (id
113), stopped reporting real values around 2022 and is deliberately
excluded. Re-verify this list periodically --
GET /api/v1/rain/?municipality=35001&ordering=-measured_on&limit=15&format=json
shows which stations currently have fresh, non-null readings.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))

BIPAD_BASE_URL = "https://bipadportal.gov.np/api/v1"
REQUEST_TIMEOUT_SECONDS = 10

RIVER_SEARCH_TERMS = {
    "devghat": "Narayani at Devghat",
    "rajaiya": "East Rapti at Rajaiya",
}

# Verified actively-reporting rain gauges inside Bharatpur Metropolitan City
# (municipality=35001) as of 2026-07-27. "Bharatpur Airport" (station 113) is
# excluded: its most recent readings are all null (sensor appears offline).
ACTIVE_RAIN_STATION_IDS: dict[int, str] = {
    167: "Rampur_AWS",
    280: "Narayani Field Office",
    622: "Meghauli",
}

# ============================================================================
# TEMPORARY -- FAKE LIVE DATA FOR SCREENSHOT/DEMO PURPOSES.
# BIPAD Portal is down as of this flag being added; the two _fetch_* functions
# below short-circuit with realistic static values instead of calling the real
# API, so GET /api/conditions/live still returns a normal 200 (not a 502) for
# a dashboard screenshot. This is meant to be short-lived -- set back to False
# (or delete this block and the two `if FAKE_LIVE_DATA_FOR_DEMO:` branches
# below) to restore real fetching. Added 2026-08-08, revert on request.
# ============================================================================
FAKE_LIVE_DATA_FOR_DEMO = False

_CACHE_TTL_SECONDS = 300  # don't hit BIPAD more than once per 5 minutes

# Keyed by station ("devghat"/"rajaiya"), NOT a single global slot -- a
# single shared slot meant two near-simultaneous requests for *different*
# stations (the frontend's page-load/Refresh flow does both via
# Promise.all, see RIVER_STATION_FIX_FRONTEND_PROMPT.md) could each write
# to the same cache entry, and whichever one wrote last silently determined
# what every subsequent call got back regardless of which station it asked
# for. Confirmed against live BIPAD data 2026-08: devghat and rajaiya have
# very different real readings (5.29m vs 1.12m at time of writing), so this
# wasn't a cosmetic bug -- callers could receive another station's water
# level mislabeled as their own, including into RiverLevelRecord history
# rows via app/routes/conditions.py's _log_live_conditions().
_river_cache: dict[str, dict[str, Any]] = {}

# Rainfall is basin-wide (not per-station), so it's cached separately from
# the river reading rather than being refetched on every call -- the
# dual-station frontend flow above calls get_live_conditions() twice per
# page load/refresh, which would otherwise mean 2x the necessary rainfall
# API calls (3 sequential BIPAD requests each) for identical data.
_rainfall_cache: dict[str, Any] = {"value": None, "fetched_at_monotonic": 0.0, "fetched_at": None}


class BipadFetchError(RuntimeError):
    """Raised when BIPAD's API is unreachable or returns unexpected data."""


def _get_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{BIPAD_BASE_URL}/{path}"
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise BipadFetchError(f"Request to {url} failed: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise BipadFetchError(f"Non-JSON response from {url}") from exc


_FAKE_RIVER_READINGS: dict[str, dict[str, Any]] = {
    # Matches the real readings already cited elsewhere in this project's docs
    # (FRONTEND_CONTEXT.md, RIVER_STATION_FIX_FRONTEND_PROMPT.md) so a
    # screenshot taken now stays consistent with what's already written down.
    "devghat": {"water_level_m": 5.29, "warning_level_m": 7.3, "danger_level_m": 9.0,
                "station_title": "Narayani at Devghat"},
    "rajaiya": {"water_level_m": 1.12, "warning_level_m": 3.3, "danger_level_m": 3.7,
                "station_title": "East Rapti at Rajaiya"},
}


def _fetch_latest_river_reading(station_key: str) -> dict[str, Any]:
    if station_key not in RIVER_SEARCH_TERMS:
        raise ValueError(f"Unknown river station_key '{station_key}'")

    if FAKE_LIVE_DATA_FOR_DEMO:  # see the flag's module-level comment -- temporary
        fake = _FAKE_RIVER_READINGS[station_key]
        return {**fake, "measured_on": datetime.now(NEPAL_TZ).isoformat()}

    data = _get_json(
        "river/",
        {
            "search": RIVER_SEARCH_TERMS[station_key],
            "ordering": "-water_level_on",
            "limit": 1,
            "format": "json",
            "fields": "id,title,water_level,danger_level,warning_level,water_level_on,status",
        },
    )
    results = data.get("results") or []
    if not results:
        raise BipadFetchError(f"No river readings returned for '{station_key}'")

    reading = results[0]
    water_level = reading.get("waterLevel")
    if water_level is None or water_level < -1000:  # BIPAD uses ~-9999985 as a sentinel for bad readings
        raise BipadFetchError(f"'{station_key}' latest reading is invalid/sentinel: {water_level}")

    return {
        "water_level_m": float(water_level),
        "warning_level_m": reading.get("warningLevel"),
        "danger_level_m": reading.get("dangerLevel"),
        "measured_on": reading.get("waterLevelOn"),
        "station_title": reading.get("title"),
    }


def _fetch_latest_rainfall_averages() -> dict[str, Any]:
    if FAKE_LIVE_DATA_FOR_DEMO:  # see the flag's module-level comment -- temporary
        return {
            "rainfall_1h_mm": 0.0,
            "rainfall_3h_mm": 0.0,
            "rainfall_6h_mm": 0.0,
            "rainfall_12h_mm": 1.2,
            "rainfall_24h_mm": 12.4,
            "rain_stations_used": ["Rampur_AWS", "Narayani Field Office", "Meghauli"],
            "measured_on": datetime.now(NEPAL_TZ).isoformat(),
        }

    per_interval_values: dict[int, list[float]] = {1: [], 3: [], 6: [], 12: [], 24: []}
    stations_used: list[str] = []
    latest_measured_on: str | None = None

    for station_id, station_name in ACTIVE_RAIN_STATION_IDS.items():
        data = _get_json(
            "rain/",
            {"station": station_id, "ordering": "-measured_on", "limit": 1, "format": "json"},
        )
        results = data.get("results") or []
        if not results:
            continue

        reading = results[0]
        measured_on = reading.get("measuredOn")
        if measured_on and (latest_measured_on is None or measured_on > latest_measured_on):
            latest_measured_on = measured_on

        has_value = False
        for average in reading.get("averages", []):
            interval = average.get("interval")
            value = average.get("value")
            if interval in per_interval_values and value is not None:
                per_interval_values[interval].append(float(value))
                has_value = True

        if has_value:
            stations_used.append(station_name)

    if not stations_used:
        raise BipadFetchError("No active rain stations returned usable readings")

    averaged = {
        interval: (round(sum(values) / len(values), 2) if values else 0.0)
        for interval, values in per_interval_values.items()
    }

    return {
        "rainfall_1h_mm": averaged[1],
        "rainfall_3h_mm": averaged[3],
        "rainfall_6h_mm": averaged[6],
        "rainfall_12h_mm": averaged[12],
        "rainfall_24h_mm": averaged[24],
        "rain_stations_used": stations_used,
        "measured_on": latest_measured_on,
    }


def _get_river_reading_cached(station_key: str) -> tuple[dict[str, Any], str]:
    """Returns (reading, fetched_at_iso), from this station's own cache slot
    -- see _river_cache's module-level comment for why this must be keyed
    per station rather than one shared slot."""
    now = time.monotonic()
    cached = _river_cache.get(station_key)
    if cached is not None and (now - cached["fetched_at_monotonic"]) < _CACHE_TTL_SECONDS:
        return cached["value"], cached["fetched_at"]

    reading = _fetch_latest_river_reading(station_key)
    fetched_at = datetime.now(NEPAL_TZ).isoformat()
    _river_cache[station_key] = {
        "value": reading,
        "fetched_at_monotonic": now,
        "fetched_at": fetched_at,
    }
    return reading, fetched_at


def _get_rainfall_averages_cached() -> tuple[dict[str, Any], str]:
    """Returns (averages, fetched_at_iso). Shared across all stations --
    rainfall doesn't vary by river gauge, see _rainfall_cache's comment."""
    now = time.monotonic()
    if _rainfall_cache["value"] is not None and (
        now - _rainfall_cache["fetched_at_monotonic"]
    ) < _CACHE_TTL_SECONDS:
        return _rainfall_cache["value"], _rainfall_cache["fetched_at"]

    averages = _fetch_latest_rainfall_averages()
    fetched_at = datetime.now(NEPAL_TZ).isoformat()
    _rainfall_cache["value"] = averages
    _rainfall_cache["fetched_at_monotonic"] = now
    _rainfall_cache["fetched_at"] = fetched_at
    return averages, fetched_at


def get_live_conditions(primary_river_station: str = "devghat") -> dict[str, Any]:
    """Fetch current rainfall (averaged across active Bharatpur-area gauges)
    and river level (from `primary_river_station`) from BIPAD. Each of the
    two independently cached for _CACHE_TTL_SECONDS (see
    _get_river_reading_cached/_get_rainfall_averages_cached -- keyed
    separately, NOT one shared cache entry for the whole result, which used
    to let two concurrent different-station requests clobber each other).
    Raises BipadFetchError if BIPAD can't be reached -- callers should
    surface that as a clear error, not silently fall back to zeros (the
    frontend needs to distinguish "unavailable" from "calm").
    """
    river, river_fetched_at = _get_river_reading_cached(primary_river_station)
    rainfall, rainfall_fetched_at = _get_rainfall_averages_cached()

    return {
        "rainfall_1h_mm": rainfall["rainfall_1h_mm"],
        "rainfall_3h_mm": rainfall["rainfall_3h_mm"],
        "rainfall_6h_mm": rainfall["rainfall_6h_mm"],
        "rainfall_12h_mm": rainfall["rainfall_12h_mm"],
        "rainfall_24h_mm": rainfall["rainfall_24h_mm"],
        "water_level_m": river["water_level_m"],
        "warning_level_m": river["warning_level_m"],
        "danger_level_m": river["danger_level_m"],
        "river_station_id": primary_river_station,
        "rain_stations_used": rainfall["rain_stations_used"],
        "source": "bipad",
        # The more recent of the two components' own fetch times (ISO 8601
        # with a fixed offset sorts correctly as a plain string comparison)
        # -- river and rainfall can now be cached independently, so a single
        # blended "fetched_at" needs to reflect whichever is actually
        # fresher/staler, not just "now" (which would overstate freshness
        # on a cache hit).
        "fetched_at": max(river_fetched_at, rainfall_fetched_at),
    }
