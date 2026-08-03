"""
acquire_chirps_via_earthengine.py

PURPOSE
-------
Downloads CHIRPS v3 daily rainfall (1993-01-01 to 2020-12-31) at your 7
rainfall station locations, via Google Earth Engine, and writes it out
in EXACTLY the schema 06_chirps_glofas_historical_backfill.py expects:
    date, station_id, rainfall_24h_mm

RUN THIS LOCALLY, NOT IN THE CLAUDE SANDBOX.
earthengine.google.com/googleapis.com are not reachable from this
sandbox's network allowlist.

ONE-TIME SETUP (run once, ever)
---------------------------------
1. Sign up for a free Earth Engine account (if you don't have one):
   https://signup.earthengine.google.com/
   (Approval is usually near-instant for research/student accounts.)

2. Install the Earth Engine Python API:
   pip install earthengine-api geopandas pandas --break-system-packages

3. Authenticate once (opens a browser, one-time):
   python -c "import ee; ee.Authenticate()"
   This caches a token locally -- you won't need to repeat this on future runs.

4. Edit CLOUD_PROJECT below to your own Earth Engine-enabled Google Cloud
   project ID (Earth Engine requires this now -- create a free one at
   https://console.cloud.google.com/ if needed, then enable the Earth
   Engine API for it).

WHY EARTH ENGINE (NOT A DIRECT CHC DOWNLOAD)
-----------------------------------------------
Earth Engine already has the full CHIRPS v3 archive hosted and lets you
extract values at specific points server-side, so you don't need to
download and process global daily GeoTIFFs yourself (each daily CHIRPS
v3 global file is a non-trivial download, x 27 years x 365 days). This
is the standard, recommended approach for point-extraction use cases.

INPUT
------
rainfall_stations.csv  -- your existing 7-station file, needs
                            station_id, latitude, longitude columns

OUTPUT
------
chirps_daily_1993_2020.csv
    columns: date, station_id, rainfall_24h_mm
    -- this is the exact file 06_chirps_glofas_historical_backfill.py
       expects as CHIRPS_PATH.
"""

import sys
import pandas as pd

CLOUD_PROJECT = "winged-yeti-345514"   # <-- EDIT THIS to your own project ID
STATIONS_CSV = "rainfall_stations.csv"
OUT_PATH = "chirps_daily_1993_2020.csv"
START_DATE = "1993-01-01"
END_DATE = "2021-01-01"   # exclusive end, so this covers through 2020-12-31

# CHIRPS v3 daily collection ID as documented on Earth Engine's catalog.
# If Earth Engine has since renamed/moved this (check
# https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY
# for the current ID), update this string.
CHIRPS_COLLECTION_ID = "UCSB-CHG/CHIRPS/DAILY"


def main():
    try:
        import ee
    except ImportError:
        print("earthengine-api not installed. Run:")
        print("  pip install earthengine-api --break-system-packages")
        sys.exit(1)

    try:
        ee.Initialize(project=CLOUD_PROJECT)
    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        print("Things to check:")
        print("  1. Did you run `python -c \"import ee; ee.Authenticate()\"` at least once?")
        print("  2. Is CLOUD_PROJECT set to a real Google Cloud project ID with the")
        print("     Earth Engine API enabled?")
        sys.exit(1)

    try:
        stations = pd.read_csv(STATIONS_CSV)
    except FileNotFoundError:
        print(f"MISSING INPUT: {STATIONS_CSV}")
        sys.exit(1)

    print(f"Loaded {len(stations)} rainfall stations.")
    print(f"Pulling CHIRPS v3 daily rainfall, {START_DATE} to {END_DATE} ...")

    # Earth Engine's getInfo() aborts after ~5000 accumulated elements.
    # 1993-2020 is ~28 years x ~365 daily images per station = ~10,000+
    # images -- comfortably over that limit in one call. Chunk by year
    # (365 images/year, safely under 5000) and concatenate client-side.
    year_ranges = [(f"{y}-01-01", f"{y+1}-01-01") for y in range(1993, 2021)]

    def extract_year(pt, station_id, year_start, year_end):
        chirps_year = ee.ImageCollection(CHIRPS_COLLECTION_ID).filterDate(year_start, year_end)

        def extract(image):
            value = image.reduceRegion(
                reducer=ee.Reducer.first(), geometry=pt, scale=5566  # ~0.05deg native res
            ).get("precipitation")
            return ee.Feature(None, {
                "date": image.date().format("YYYY-MM-dd"),
                "rainfall_24h_mm": value,
            })

        fc = chirps_year.map(extract)
        return fc.getInfo()

    all_rows = []
    for _, row in stations.iterrows():
        station_id = str(row["station_id"])
        pt = ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])])
        print(f"  Extracting time series for station {station_id} "
              f"({row['latitude']:.4f}, {row['longitude']:.4f}) ...")

        station_rows = []
        for year_start, year_end in year_ranges:
            year_label = year_start[:4]
            try:
                data = extract_year(pt, station_id, year_start, year_end)
            except Exception as e:
                print(f"    ERROR extracting station {station_id}, year {year_label}: {e}")
                print("    If this STILL hits the element limit, the collection may include")
                print("    more than one image per day for some dates -- try splitting by")
                print("    month instead of by year (change year_ranges to monthly chunks).")
                continue

            for feat in data["features"]:
                props = feat["properties"]
                station_rows.append({
                    "date": props["date"],
                    "station_id": station_id,
                    "rainfall_24h_mm": props["rainfall_24h_mm"],
                })
            print(f"    {year_label}: {len(data['features'])} days extracted")

        all_rows.extend(station_rows)
        print(f"  Station {station_id} total: {len(station_rows)} days")

    if not all_rows:
        print("No data extracted -- check errors above.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df):,} rows to {OUT_PATH}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stations covered: {sorted(df['station_id'].unique())}")
    print(f"Missing values: {df['rainfall_24h_mm'].isna().sum()}")
    print("\nNext step: run 06_chirps_glofas_historical_backfill.py")


if __name__ == "__main__":
    main()