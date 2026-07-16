import json
import time
import requests
import pandas as pd
from urllib.parse import urlparse, parse_qsl

API_URL = "https://bipadportal.gov.np/api/v1/river/?title=Narayani%20at%20Devghat&historical=true&format=json&water_level_on__gt=2026-07-04T00%3A00%3A00%2B05%3A45&water_level_on__lt=2026-07-07T23%3A59%3A59%2B05%3A45&fields=id%2Ccreated_on%2Ctitle%2Cbasin%2Cpoint%2Cimage%2Cwater_level%2Cdanger_level%2Cwarning_level%2Cwater_level_on%2Cstatus%2Csteady%2Cdescription%2Cstation"

OUTPUT_CSV = "narayani_devghat_paginated.csv"
OUTPUT_JSON = "narayani_devghat_paginated_raw.json"


def prepare_url(api_url, limit=500):
    parsed = urlparse(api_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    params = dict(parse_qsl(parsed.query))

    # Important: replace limit=-1 with real pagination
    params["limit"] = str(limit)
    params["offset"] = "0"

    return base_url, params


def classify_level(water_level, warning_level, danger_level):
    if water_level is None:
        return "UNKNOWN"

    if danger_level is not None and water_level >= danger_level:
        return "DANGER"

    if warning_level is not None and water_level >= warning_level:
        return "WARNING"

    return "NORMAL"


def flatten_record(row):
    point = row.get("point") or {}
    coordinates = point.get("coordinates") or [None, None]

    water_level = row.get("waterLevel")
    warning_level = row.get("warningLevel")
    danger_level = row.get("dangerLevel")

    return {
        "id": row.get("id"),
        "created_on": row.get("createdOn"),
        "title": row.get("title"),
        "basin": row.get("basin"),
        "longitude": coordinates[0],
        "latitude": coordinates[1],
        "water_level": water_level,
        "warning_level": warning_level,
        "danger_level": danger_level,
        "water_level_on": row.get("waterLevelOn"),
        "status": row.get("status"),
        "steady": row.get("steady"),
        "description": row.get("description"),
        "station": row.get("station"),
        "image": row.get("image"),
        "risk_status": classify_level(
            water_level,
            warning_level,
            danger_level
        )
    }


def scrape_multiple_pages(api_url, limit=500, delay=0.3):
    base_url, params = prepare_url(api_url, limit=limit)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    })

    all_records = []
    offset = 0

    while True:
        params["offset"] = str(offset)

        print(f"Fetching page: offset={offset}, limit={limit}")

        response = session.get(base_url, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        records = data.get("results", [])

        print(f"Records received: {len(records)}")

        if not records:
            break

        all_records.extend(records)

        # Do not trust count because BIPAD may return a huge placeholder count.
        # Stop when this page has fewer records than the limit.
        if len(records) < limit:
            break

        offset += limit
        time.sleep(delay)

    return all_records


records = scrape_multiple_pages(API_URL, limit=500)

print("Total scraped:", len(records))

# Save raw JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

# Convert to dataframe
flat_records = [flatten_record(row) for row in records]
df = pd.DataFrame(flat_records)

if not df.empty:
    df["water_level_on"] = pd.to_datetime(df["water_level_on"], errors="coerce")
    df["created_on"] = pd.to_datetime(df["created_on"], errors="coerce")

    df = df.drop_duplicates(subset=["id"])
    df = df.sort_values("water_level_on")

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Saved CSV:", OUTPUT_CSV)
print("Saved JSON:", OUTPUT_JSON)
print(df.head())