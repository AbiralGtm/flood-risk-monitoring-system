"""
Bulk download historical river-level data from BIPAD's /river/ endpoint
(found via browser Network tab) for Devghat (Narayani) and Rajaiya (East
Rapti). Unlike /rain/, this endpoint gives danger_level and warning_level
directly on EVERY record -- so we get real, authoritative thresholds for
free, including Rajaiya's, which manual searching couldn't find earlier.

Output columns already computed, ready for the rest of the pipeline:
  station_id, bipad_title, datetime, water_level_m, warning_level_m,
  danger_level_m, dhm_status, steady, river_level_score
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

BASE_URL = "http://bipadportal.gov.np/api/v1/river/"
OUT_CSV = "river_level_hourly_multistation.csv"

# Confirmed / likely BIPAD titles -- Devghat confirmed directly from your
# found URL. Rajaiya's title is from the DHM river-watch dropdown list
# found earlier ("East Rapti at Rajaiya") -- verify this is exact once
# the script runs; if it returns 0 records, the title string is off.
STATIONS = [
    {"station_id": "RV01", "bipad_title": "Narayani at Devghat"},
    {"station_id": "RV02", "bipad_title": "East Rapti at Rajaiya"},
]

START_DATE = datetime(2021, 5, 1)
END_DATE = datetime(2026, 7, 21)
CHUNK_MONTHS = 1
SLEEP_BETWEEN_CALLS = 0.5

FIELDS = ("id,created_on,title,basin,point,water_level,danger_level,"
          "warning_level,water_level_on,status,steady,description,station")


def month_chunks(start, end, months):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=31 * months), end)
        yield cur, nxt
        cur = nxt


def fetch_chunk(title, start, end):
    params = {
        "title": title,
        "historical": "true",
        "format": "json",
        "water_level_on__gt": start.strftime("%Y-%m-%dT%H:%M:%S+05:45"),
        "water_level_on__lt": end.strftime("%Y-%m-%dT%H:%M:%S+05:45"),
        "fields": FIELDS,
        "limit": -1,
    }
    r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("results", data if isinstance(data, list) else [])


def river_level_score(water_level, warning_level, danger_level):
    if warning_level is None or danger_level is None or water_level is None:
        return None
    if warning_level == 0 or danger_level == warning_level:
        return None
    if water_level >= danger_level:
        return 1.0
    if water_level >= warning_level:
        return 0.5 + 0.5 * (water_level - warning_level) / (danger_level - warning_level)
    return max(0.0, 0.5 * water_level / warning_level)


def get_field(rec, snake_name, camel_name):
    """Try camelCase first (the pattern BIPAD actually uses for output keys),
    fall back to snake_case in case this endpoint behaves differently."""
    if camel_name in rec:
        return rec[camel_name]
    return rec.get(snake_name)


# ----------------------------------------------------------------------
# PRE-FLIGHT CHECK: fetch a handful of real records first and verify field
# names before committing to the full multi-year download. We've hit this
# exact class of bug twice now -- this catches it in seconds instead of
# after 660 API calls.
# ----------------------------------------------------------------------
print("Pre-flight check: fetching a small sample first...")
sample_records = fetch_chunk(STATIONS[0]["bipad_title"], datetime(2026, 7, 1), datetime(2026, 7, 5))
if not sample_records:
    print(f"*** No records returned at all for '{STATIONS[0]['bipad_title']}' in a "
          f"recent 4-day test window. Check the title spelling before proceeding. ***")
    raise SystemExit(1)

sample = sample_records[0]
print("Raw sample record:", sample)
test_dt = get_field(sample, "water_level_on", "waterLevelOn")
if test_dt is None:
    print("\n*** STOP: neither 'water_level_on' nor 'waterLevelOn' found in the ")
    print("actual response. Real keys present:", list(sample.keys()))
    print("Update get_field() calls below to match these exact key names, then re-run. ***")
    raise SystemExit(1)
print(f"Pre-flight check passed -- datetime field found: {test_dt}\n")

all_rows = []
for s in STATIONS:
    station_id = s["station_id"]
    title = s["bipad_title"]
    print(f"\n--- {station_id} ({title}) ---")
    station_total = 0

    for chunk_start, chunk_end in month_chunks(START_DATE, END_DATE, CHUNK_MONTHS):
        try:
            records = fetch_chunk(title, chunk_start, chunk_end)
            for rec in records:
                all_rows.append({
                    "station_id": station_id,
                    "bipad_title": rec.get("title"),
                    "datetime": get_field(rec, "water_level_on", "waterLevelOn"),
                    "created_on": get_field(rec, "created_on", "createdOn"),
                    "water_level_m": get_field(rec, "water_level", "waterLevel"),
                    "warning_level_m": get_field(rec, "warning_level", "warningLevel"),
                    "danger_level_m": get_field(rec, "danger_level", "dangerLevel"),
                    "dhm_status": rec.get("status"),
                    "steady": rec.get("steady"),
                })
            station_total += len(records)
            print(f"  {chunk_start.date()} to {chunk_end.date()}: {len(records)} records")
        except Exception as e:
            print(f"  FAILED {chunk_start.date()} to {chunk_end.date()}: {e}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"  TOTAL for {title}: {station_total} records")
    if station_total == 0:
        print(f"  *** WARNING: zero records for '{title}' -- title string is "
              f"likely wrong. Check the exact spelling on BIPAD's river-watch "
              f"page dropdown and update STATIONS in this script. ***")

df = pd.DataFrame(all_rows)
print(f"\nTotal records downloaded: {len(df):,}")

if len(df):
    missing_dt = df["datetime"].isna().sum()
    if missing_dt == len(df):
        print("*** STOP: every record has missing datetime -- field name issue, "
              "same class of bug as before. Check fields= param spelling. ***")
        df.to_csv("river_level_RAW_DEBUG.csv", index=False)
        raise SystemExit(1)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values(["station_id", "datetime"]).drop_duplicates(
        subset=["station_id", "datetime"]
    )

    df["river_level_score"] = df.apply(
        lambda r: river_level_score(r["water_level_m"], r["warning_level_m"], r["danger_level_m"]),
        axis=1,
    )

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print("\nRecords per station:")
    print(df["station_id"].value_counts())
    print("\nDate range covered:")
    print(df["datetime"].min(), "to", df["datetime"].max())

    print("\n--- Thresholds discovered directly from BIPAD (use to update river_stations.csv) ---")
    for sid, grp in df.groupby("station_id"):
        wl = grp["warning_level_m"].dropna().unique()
        dl = grp["danger_level_m"].dropna().unique()
        print(f"{sid}: warning_level_m={wl}, danger_level_m={dl}")
else:
    print("No records downloaded -- check station titles and endpoint.")