"""
STEP A — Discover BIPAD's exact station "title" strings for your 11 known
rainfall stations. We only confirmed one so far (Rampur_AWS = station 902
Rampur). BIPAD's API filters by exact title string, so we need the right
name for each station before bulk-downloading.

Strategy: query BIPAD broadly (no title filter) for a short recent window,
across both relevant basins, collect every unique (title, station_id,
coordinates) triple seen, then match each to your known stations by
nearest coordinate.

Output:
  - bipad_station_directory.csv   (everything BIPAD reports seeing)
  - bipad_title_mapping.csv       (your station_id -> matched BIPAD title,
    with match distance so you can sanity-check/correct any bad matches)
"""

import requests
import pandas as pd
import math

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

BASE_URL = "http://bipadportal.gov.np/api/v1/rain/"
RAINFALL_STATIONS_CSV = "rainfall_stations.csv"
OUT_DIRECTORY_CSV = "bipad_station_directory.csv"
OUT_MAPPING_CSV = "bipad_title_mapping.csv"

# A short, recent window is enough to see every currently-active station
# report at least once (most report hourly).
QUERY_PARAMS_TEMPLATE = {
    "is_hourly": "true",
    "measured_on__gt": "2026-07-15T00:00:00+05:45",
    "measured_on__lt": "2026-07-16T00:00:00+05:45",
    "fields": "id,title,basin,measuredOn,point,station",
    "limit": -1,
}

# ----------------------------------------------------------------------
# 1. Pull everything BIPAD has for this window (no title filter)
# ----------------------------------------------------------------------
resp = requests.get(BASE_URL, headers=HEADERS, params=QUERY_PARAMS_TEMPLATE, timeout=30)
resp.raise_for_status()
data = resp.json()
results = data.get("results", data if isinstance(data, list) else [])

rows = []
for r in results:
    coords = r.get("point", {}).get("coordinates", [None, None])
    rows.append({
        "bipad_station_id": r.get("station"),
        "bipad_title": r.get("title"),
        "basin": r.get("basin"),
        "longitude": coords[0],
        "latitude": coords[1],
    })

directory = pd.DataFrame(rows).drop_duplicates(subset=["bipad_station_id", "bipad_title"])
directory.to_csv(OUT_DIRECTORY_CSV, index=False)
print(f"Found {len(directory)} unique BIPAD stations reporting in this window.")
print(directory[["bipad_title", "basin", "latitude", "longitude"]].to_string(index=False))

# ----------------------------------------------------------------------
# 2. Match each of your known rainfall stations to the nearest BIPAD entry
# ----------------------------------------------------------------------
known = pd.read_csv(RAINFALL_STATIONS_CSV)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

matches = []
for _, k in known.iterrows():
    best = None
    best_dist = float("inf")
    for _, d in directory.iterrows():
        if pd.isna(d["latitude"]) or pd.isna(d["longitude"]):
            continue
        dist = haversine_km(k["latitude"], k["longitude"], d["latitude"], d["longitude"])
        if dist < best_dist:
            best_dist = dist
            best = d
    matches.append({
        "station_id": k["station_id"],
        "station_name": k["station_name"],
        "matched_bipad_title": best["bipad_title"] if best is not None else None,
        "matched_bipad_station_id": best["bipad_station_id"] if best is not None else None,
        "match_distance_km": round(best_dist, 2) if best is not None else None,
    })

mapping = pd.DataFrame(matches)
mapping.to_csv(OUT_MAPPING_CSV, index=False)
print(f"\nSaved mapping to {OUT_MAPPING_CSV}:")
print(mapping.to_string(index=False))
print("\nCHECK match_distance_km for each row -- anything above ~2km is a")
print("suspicious match (could be the wrong station, or your station simply")
print("isn't in BIPAD's system under a different name/network). Verify")
print("manually before bulk downloading.")
