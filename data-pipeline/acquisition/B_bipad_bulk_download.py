"""
STEP B — Bulk download rainfall history from BIPAD for every matched
station, across your full 2021-05 to 2026-07 window. Chunked by month
(60 chunks per station) to stay well within request size/timeout limits
-- adjust CHUNK_MONTHS if you find larger windows work fine.

Uses the pre-aggregated `averages` field directly (no rolling-window
math needed -- BIPAD already computed 1h/3h/6h/12h/24h accumulations
per reading).

Input:
  - bipad_title_mapping.csv   (from step A -- verify matches first!)

Output:
  - rainfall_hourly_multistation.csv
    columns: station_id, bipad_title, datetime, rainfall_1h_mm,
             rainfall_3h_mm, rainfall_6h_mm, rainfall_12h_mm,
             rainfall_24h_mm, rainfall_score, dhm_status
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

BASE_URL = "http://bipadportal.gov.np/api/v1/rain/"
MAPPING_CSV = "bipad_title_mapping.csv"
OUT_CSV = "rainfall_hourly_multistation.csv"

START_DATE = datetime(2021, 5, 1)
END_DATE = datetime(2026, 7, 17)
CHUNK_MONTHS = 1          # pull one month at a time per station
SLEEP_BETWEEN_CALLS = 0.5  # be polite to the server -- avoid hammering it

FIELDS = "id,measured_on,title,basin,averages,status,station"


def month_chunks(start, end, months):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=31 * months), end)
        yield cur, nxt
        cur = nxt


def fetch_chunk(title, start, end):
    params = {
        "title": title,
        "is_hourly": "true",
        "measured_on__gt": start.strftime("%Y-%m-%dT%H:%M:%S+05:45"),
        "measured_on__lt": end.strftime("%Y-%m-%dT%H:%M:%S+05:45"),
        "fields": FIELDS,
        "limit": -1,
    }
    r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("results", data if isinstance(data, list) else [])


def parse_record(rec, station_id):
    measured_on = rec.get("measuredOn")
    averages = {a["interval"]: a["value"] for a in rec.get("averages", [])}
    return {
        "station_id": station_id,
        "bipad_title": rec.get("title"),
        "datetime": measured_on,
        "rainfall_1h_mm": averages.get(1),
        "rainfall_3h_mm": averages.get(3),
        "rainfall_6h_mm": averages.get(6),
        "rainfall_12h_mm": averages.get(12),
        "rainfall_24h_mm": averages.get(24),
        "dhm_status": rec.get("status"),
    }


mapping = pd.read_csv(MAPPING_CSV)
mapping = mapping.dropna(subset=["matched_bipad_title"])
print(f"Downloading for {len(mapping)} matched stations...")

all_rows = []
for _, row in mapping.iterrows():
    station_id = row["station_id"]
    title = row["matched_bipad_title"]
    print(f"\n--- {station_id} ({title}) ---")

    for chunk_start, chunk_end in month_chunks(START_DATE, END_DATE, CHUNK_MONTHS):
        try:
            records = fetch_chunk(title, chunk_start, chunk_end)
            for rec in records:
                all_rows.append(parse_record(rec, station_id))
            print(f"  {chunk_start.date()} to {chunk_end.date()}: {len(records)} records")
        except Exception as e:
            print(f"  FAILED {chunk_start.date()} to {chunk_end.date()}: {e}")
        time.sleep(SLEEP_BETWEEN_CALLS)

df = pd.DataFrame(all_rows)
print(f"\nTotal records downloaded: {len(df):,}")

if len(df):
    missing_dt = df["datetime"].isna().sum()
    print(f"Records with missing datetime BEFORE parsing: {missing_dt:,} out of {len(df):,}")
    if missing_dt == len(df):
        print("\n*** STOP: every single record has a missing/empty datetime value. ***")
        print("This means the API response for bulk/chunked requests doesn't carry")
        print("'measuredOn' the way the earlier single-day sample did. Run")
        print("diagnose_bulk_response.py first to see the raw record structure")
        print("before re-running this full download -- otherwise you'll get the")
        print("same 1-row-per-station collapse again.")
        df.to_csv("rainfall_hourly_multistation_RAW_DEBUG.csv", index=False)
        print("Saved unparsed raw data to rainfall_hourly_multistation_RAW_DEBUG.csv")
        print("for inspection instead of overwriting your real output file.")
        raise SystemExit(1)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    still_bad = df["datetime"].isna().sum()
    if still_bad:
        print(f"WARNING: {still_bad:,} records failed datetime parsing (kept as NaT) "
              f"-- these will be dropped, not silently deduplicated away.")
        df = df.dropna(subset=["datetime"])

    df = df.sort_values(["station_id", "datetime"]).drop_duplicates(
        subset=["station_id", "datetime"]
    )

    # rainfall_score, same formula used throughout the rest of the pipeline
    df["rainfall_score"] = np.clip(
        np.maximum.reduce([
            df["rainfall_1h_mm"].fillna(0) / 60,
            df["rainfall_3h_mm"].fillna(0) / 80,
            df["rainfall_6h_mm"].fillna(0) / 100,
            df["rainfall_12h_mm"].fillna(0) / 120,
            df["rainfall_24h_mm"].fillna(0) / 140,
        ]), 0, 1,
    )

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print("\nRecords per station:")
    print(df["station_id"].value_counts())
    print("\nDate range actually covered:")
    print(df["datetime"].min(), "to", df["datetime"].max())
else:
    print("No records downloaded -- check the mapping file and API responses above.")