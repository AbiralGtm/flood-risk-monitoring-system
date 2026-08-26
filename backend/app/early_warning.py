"""
early_warning.py

Independent early-warning layer, sitting NEXT TO the trained model, not
inside it. Two pure functions, no model retraining, no feature changes.

Rule 1: rate-of-rise on the grid's own river gauge -- catches "level hasn't
         crossed threshold yet, but is climbing fast."
Rule 2: catchment-wide rainfall check -- catches "heavy rain is already
         happening upstream, before Devghat's own reading responds."

Both rules use data you already collect live; nothing new to source.
"""

from datetime import datetime, timedelta
from typing import Optional


def check_rate_of_rise(
    readings: list[tuple[datetime, float]],
    rate_threshold_m_per_hr: float = 0.5,
) -> tuple[bool, Optional[float]]:
    """
    readings: list of (timestamp, water_level_m) for ONE station, most recent
              readings first or last -- any order, sorted internally.
    Returns (flag, rate_m_per_hr). flag=True if the level is climbing faster
    than rate_threshold_m_per_hr, computed from the earliest-to-latest
    reading pair available within the last hour.
    """
    if len(readings) < 2:
        return False, None

    readings = sorted(readings, key=lambda r: r[0])
    latest_time, latest_level = readings[-1]
    cutoff = latest_time - timedelta(hours=1)

    # earliest reading still within the last hour (falls back to the
    # oldest available reading if none exist exactly 1hr back)
    candidates = [r for r in readings if r[0] >= cutoff]
    if len(candidates) < 2:
        earliest_time, earliest_level = readings[0]
    else:
        earliest_time, earliest_level = candidates[0]

    elapsed_hr = (latest_time - earliest_time).total_seconds() / 3600
    if elapsed_hr <= 0:
        return False, None

    rate = (latest_level - earliest_level) / elapsed_hr
    return rate >= rate_threshold_m_per_hr, round(rate, 3)


def check_catchment_rainfall(
    station_readings_mm: dict[str, float],
    threshold_mm: float = 140.0,
) -> tuple[bool, Optional[str]]:
    """
    station_readings_mm: {station_name: rainfall_24h_mm} for ALL stations in
                         a river's catchment, not just the grid's assigned one.
    Returns (flag, triggering_station_name_or_None).
    Threshold matches the existing, DHM-verified 140mm/24h figure already
    used elsewhere in the pipeline -- no new number introduced.
    """
    if not station_readings_mm:
        return False, None
    worst_station = max(station_readings_mm, key=station_readings_mm.get)
    worst_value = station_readings_mm[worst_station]
    return worst_value >= threshold_mm, (worst_station if worst_value >= threshold_mm else None)


def get_early_warning(
    river_readings: list[tuple[datetime, float]],
    catchment_rainfall_mm: dict[str, float],
    rate_threshold_m_per_hr: float = 0.5,
    rainfall_threshold_mm: float = 140.0,
) -> dict:
    """
    Combines both rules into one response block, ready to attach alongside
    the model's normal prediction output. Does not alter the model's own
    risk_class or probabilities in any way.
    """
    rate_flag, rate_value = check_rate_of_rise(river_readings, rate_threshold_m_per_hr)
    rain_flag, rain_station = check_catchment_rainfall(catchment_rainfall_mm, rainfall_threshold_mm)

    reasons = []
    if rate_flag:
        reasons.append(f"river level rising at {rate_value} m/hr (threshold {rate_threshold_m_per_hr} m/hr)")
    if rain_flag:
        reasons.append(f"{rain_station} station reporting {catchment_rainfall_mm[rain_station]}mm/24h "
                        f"(threshold {rainfall_threshold_mm}mm)")

    return {
        "early_warning": rate_flag or rain_flag,
        "early_warning_reasons": reasons,
    }
