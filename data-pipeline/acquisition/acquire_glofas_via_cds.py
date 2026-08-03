"""
acquire_glofas_via_cds.py

PURPOSE
-------
Downloads GloFAS-ERA5 historical discharge reanalysis (1993-01-01 to
2020-12-31) at the grid cells nearest Devghat (Narayani) and Rajaiya
(Rapti), via the Copernicus Climate Data Store, and writes it out in
EXACTLY the schema 06_chirps_glofas_historical_backfill.py expects:
    date, gauge_id, discharge_m3s

RUN THIS LOCALLY, NOT IN THE CLAUDE SANDBOX.
cds.climate.copernicus.eu is not reachable from this sandbox's network
allowlist.

ONE-TIME SETUP (run once, ever)
---------------------------------
IMPORTANT: GloFAS/CEMS-Floods data has MOVED off the general Climate
Data Store onto a SEPARATE service: the CEMS Early Warning Data Store
(EWDS) at https://ewds.climate.copernicus.eu -- this requires its OWN
account and its OWN API key, different from a general CDS account (the
"cds.climate.copernicus.eu" one used for most other Copernicus datasets
like ERA5). If you already have a CDS account for something else, that
key will NOT work here -- you need a separate EWDS one.

1. Register (free) at https://ewds.climate.copernicus.eu/ (top-right
   Login/Register).

2. After logging in, go to your EWDS profile page and copy your API key.

3. Create ~/.ewdsapirc (note: NOT .cdsapirc -- a distinct filename so it
   doesn't collide with any general CDS credentials you might also have)
   with:
     url: https://ewds.climate.copernicus.eu/api
     key: <your-ewds-key-here>

4. Install the client:
   pip install cdsapi --break-system-packages
   (same "cdsapi" package works against both CDS and EWDS -- what
   differs is the url/key you point it at, which this script reads
   from ~/.ewdsapirc explicitly rather than relying on the default
   ~/.cdsapirc that the library normally auto-loads.)

5. On the dataset's EWDS page, you must click "Accept Terms" for that
   specific dataset at least once via the web UI before API downloads
   will work: https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical?tab=download
   This is a one-time per-dataset requirement, separate from account
   registration -- API calls fail cryptically if skipped.

WHY THIS DOWNLOADS A SMALL AREA, NOT JUST TWO POINTS
--------------------------------------------------------
GloFAS-ERA5 is distributed as gridded GRIB/NetCDF, not point time series --
there's no direct "give me discharge at this lat/lon" API call. This
script requests a small bounding box around each gauge, downloads the
file, then extracts the nearest grid cell's time series locally.

INPUT
------
river_stations.csv -- your existing file, needs station_id, latitude,
                        longitude columns (Devghat=RV01, Rajaiya=RV02)

OUTPUT
------
glofas_discharge_1993_2020.csv
    columns: date, gauge_id, discharge_m3s
    -- this is the exact file 06_chirps_glofas_historical_backfill.py
       expects as GLOFAS_PATH.
"""

import sys
import os
import pandas as pd
import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None  # checked explicitly in main() with a friendly error message

RIVER_STATIONS_CSV = "river_stations.csv"
OUT_PATH = "glofas_discharge_1993_2020.csv"
YEARS = [str(y) for y in range(1993, 2021)]
BBOX_HALF_DEGREE = 0.15   # ~15-17km half-width box around each gauge.
                           # Area does NOT appear to affect request "cost"
                           # when using the confirmed grib2+unarchived
                           # format (a live-form test with a much larger
                           # area than this still cost exactly 1/day, no
                           # area penalty) -- the earlier area-based cost
                           # theory was specific to the "zip" download
                           # format's apparent per-file overhead, now
                           # replaced. This size just needs to comfortably
                           # contain the nearest GloFAS grid cell(s).

EWDS_CREDENTIALS_PATH = os.path.expanduser("~/.ewdsapirc")

# ----------------------------------------------------------------------
# CONFIRMED LIVE REQUEST FORMAT (obtained directly from the EWDS
# "Show API request" form output, not from documentation/tutorials --
# this is what actually works right now):
#   - field names are "year"/"month"/"day", NOT "hyear"/"hmonth"/"hday"
#     (older tutorials use the h-prefixed names -- those are stale)
#   - "timespan": ["time_mean"] is REQUIRED and was missing before
#   - variable is "average_river_discharge_in_the_last_24_hours", NOT
#     "river_discharge_in_the_last_24_hours" -- also renamed
#   - "data_format": "grib2", "download_format": "unarchived" -- CONFIRMED
#     from the live EWDS form's "Show API request" output on 2026-08-01.
#     This is now known to matter a lot: "download_format": "zip" (used in
#     an earlier version of this script) appears to carry a heavy per-file
#     cost overhead -- a single year (372 day-month combos) with "zip"
#     cost 17,532 (~47x per unit), while the SAME live form confirmed 3
#     full years (1,095 real calendar days) cost exactly 1,095 with
#     grib2+unarchived -- i.e. cost = 1 per day, no per-file penalty, no
#     area penalty. Use grib2+unarchived, NOT zip, for any real-size pull.
# ----------------------------------------------------------------------
BISECTION_TEST = False   # set True again if this new format still fails,
                         # to isolate size vs. value issues as before
TEST_YEAR = "2005"
TEST_MONTH = "01"
TEST_DAY = "01"


def open_glofas_file(file_path):
    """
    With download_format='unarchived' and data_format='grib2', retrieve()
    downloads a single GRIB2 file directly -- no zip unwrapping needed
    (unlike the earlier 'zip' format this script used before switching).
    GRIB2 needs the cfgrib engine for xarray (pip install cfgrib eccodes).
    """
    try:
        return xr.open_dataset(file_path, engine="cfgrib")
    except Exception as e:
        print(f"    Failed to open as GRIB2 with cfgrib engine: {e}")
        print("    If cfgrib/eccodes aren't installed: "
              "pip install cfgrib eccodes --break-system-packages")
        raise


def is_cost_limit_error(exc):
    """
    Detects EWDS's 'Request too large. Cost X over Y limit' error
    specifically, so the adaptive splitter only retries-smaller for THIS
    known, expected error -- not for genuine auth/network/other failures,
    which should surface immediately rather than trigger pointless splitting.
    """
    msg = str(exc).lower()
    return "request too large" in msg or ("cost" in msg and "limit" in msg)


def fetch_glofas_chunk(client, base_request, area, months, days, label, depth=0, max_depth=8):
    """
    Retrieves one gauge-year's data, adaptively splitting into smaller
    month/day sub-requests whenever EWDS rejects the request for
    exceeding its cost limit. This discovers the right chunk size
    empirically rather than assuming one -- the exact cost formula
    (apparently ~timesteps x area-grid-cells) isn't documented anywhere
    reliable, so guessing a fixed chunk size risks yet another failed
    round-trip. Splits months first (fewer, larger sub-requests), then
    days once down to a single month, bottoming out at single-day
    requests if truly necessary.

    Returns a list of (extracted_data_path) strings, one per successful
    sub-request -- may be more than one if splitting was needed.
    """
    if depth > max_depth:
        raise RuntimeError(
            f"Gave up splitting {label} after {max_depth} levels -- even a "
            f"single day request is being rejected. This means the cost "
            f"limit issue isn't about time-range size at all; check the "
            f"area size, variable count, or credentials instead."
        )

    file_path = f"glofas_{label}_{depth}_{len(months)}m_{len(days)}d.grib2"
    request = dict(base_request)
    request["month"] = months
    request["day"] = days
    request["area"] = area

    try:
        result = client.retrieve("cems-glofas-historical", request)
        result.download(file_path)
        return [file_path]
    except Exception as e:
        if not is_cost_limit_error(e):
            raise  # a genuine different error -- surface it, don't mask it by splitting

        print(f"    Cost limit hit at {len(months)} month(s) x {len(days)} day(s) "
              f"(depth {depth}) -- splitting smaller and retrying...")

        if len(months) > 1:
            mid = len(months) // 2
            left = fetch_glofas_chunk(client, base_request, area, months[:mid], days,
                                       label, depth + 1, max_depth)
            right = fetch_glofas_chunk(client, base_request, area, months[mid:], days,
                                        label, depth + 1, max_depth)
            return left + right
        elif len(days) > 1:
            mid = len(days) // 2
            left = fetch_glofas_chunk(client, base_request, area, months, days[:mid],
                                       label, depth + 1, max_depth)
            right = fetch_glofas_chunk(client, base_request, area, months, days[mid:],
                                        label, depth + 1, max_depth)
            return left + right
        else:
            raise RuntimeError(
                f"Cost limit exceeded even for a single month/single day request "
                f"for {label} -- this points to the area being too large still, "
                f"or a per-request overhead unrelated to time range."
            )


def load_ewds_credentials(path):
    """
    Parses a simple `key: value` style file (same format as .cdsapirc)
    without relying on cdsapi's own file-discovery, since we deliberately
    use a different filename to avoid colliding with any general CDS
    credentials the user might already have configured.
    """
    if not os.path.exists(path):
        return None, None
    creds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            creds[key.strip()] = value.strip()
    return creds.get("url"), creds.get("key")


def main():
    try:
        import cdsapi
    except ImportError:
        print("cdsapi not installed. Run:")
        print("  pip install cdsapi --break-system-packages")
        sys.exit(1)

    if xr is None:
        print("xarray not installed (needed to read the downloaded GRIB2 files). Run:")
        print("  pip install xarray cfgrib eccodes --break-system-packages")
        sys.exit(1)

    ewds_url, ewds_key = load_ewds_credentials(EWDS_CREDENTIALS_PATH)
    if not ewds_url or not ewds_key:
        print(f"MISSING: {EWDS_CREDENTIALS_PATH} not found or incomplete.")
        print("See the setup instructions in this script's docstring -- GloFAS/CEMS-Floods")
        print("data requires a SEPARATE EWDS account and API key, different from a")
        print("general Climate Data Store (CDS) account. Register at:")
        print("  https://ewds.climate.copernicus.eu/")
        sys.exit(1)

    try:
        stations = pd.read_csv(RIVER_STATIONS_CSV)
    except FileNotFoundError:
        print(f"MISSING INPUT: {RIVER_STATIONS_CSV}")
        sys.exit(1)

    # Match the same active_in_pipeline filter 00_assign_stations_to_grid.py
    # already applies -- no point burning EWDS quota/time downloading
    # discharge for gauges your pipeline has already excluded (e.g.
    # Manahari/Lothar, documented as pre-confluence tributaries, not
    # mainstem Narayani/Rapti).
    if "active_in_pipeline" in stations.columns:
        excluded = stations[stations["active_in_pipeline"] != True]
        stations = stations[stations["active_in_pipeline"] == True].copy()
        if len(excluded):
            print(f"Excluding {len(excluded)} inactive gauge(s) from GloFAS pull: "
                  f"{list(excluded.get('station_name', excluded['station_id']))}")
    else:
        print("NOTE: no active_in_pipeline column found in river_stations.csv -- "
              "pulling GloFAS data for ALL rows. If this file contains excluded "
              "tributaries (e.g. Manahari/Lothar), you'll be requesting data for "
              "gauges you don't actually use.")

    print(f"Will pull GloFAS discharge for {len(stations)} gauge(s): "
          f"{list(stations['station_id'])}")

    if BISECTION_TEST:
        print("\n*** BISECTION_TEST=True: running ONE single-day request for ONE gauge ***")
        print(f"*** Testing: gauge={stations.iloc[0]['station_id']}, "
              f"{TEST_YEAR}-{TEST_MONTH}-{TEST_DAY} ***\n")
        stations = stations.iloc[[0]].copy()

    client = cdsapi.Client(url=ewds_url, key=ewds_key)
    all_rows = []

    for _, row in stations.iterrows():
        gauge_id = str(row["station_id"])
        lat, lon = float(row["latitude"]), float(row["longitude"])

        print(f"\nRequesting GloFAS-ERA5 historical discharge for gauge {gauge_id} "
              f"({lat:.4f}, {lon:.4f}), {YEARS[0]}-{YEARS[-1]} ...")
        print("  Looping one year per request -- a single request spanning all "
              "28 years was rejected as an invalid parameter combination, most "
              "likely a request-size limit reported this way rather than as an "
              "explicit size error. One-year requests match Copernicus's own")
        print("  confirmed-working tutorial examples.")

        north, south = lat + BBOX_HALF_DEGREE, lat - BBOX_HALF_DEGREE
        east, west = lon + BBOX_HALF_DEGREE, lon - BBOX_HALF_DEGREE

        gauge_rows = []
        year_list = [TEST_YEAR] if BISECTION_TEST else YEARS
        base_request = {
            "system_version": ["version_4_0"],
            "hydrological_model": ["lisflood"],
            "product_type": ["consolidated"],
            "timespan": ["time_mean"],
            "variable": ["average_river_discharge_in_the_last_24_hours"],
            "data_format": "grib2",
            "download_format": "unarchived",
        }
        area = [north, west, south, east]  # N, W, S, E

        for year in year_list:
            month_list = [TEST_MONTH] if BISECTION_TEST else [f"{m:02d}" for m in range(1, 13)]
            day_list = [TEST_DAY] if BISECTION_TEST else [f"{d:02d}" for d in range(1, 32)]
            year_request = dict(base_request)
            year_request["year"] = [year]

            try:
                data_paths = fetch_glofas_chunk(
                    client, year_request, area, month_list, day_list,
                    label=f"{gauge_id}_{year}",
                )
            except Exception as e:
                print(f"    ERROR downloading gauge {gauge_id}, year {year}: {e}")
                print("    If this format has ALSO gone stale, re-check the EWDS 'Show API")
                print("    request' output again -- Copernicus updates this periodically")
                print("    and this script's request dict needs to match it exactly.")
                continue

            year_days_extracted = 0
            for data_path in data_paths:
                try:
                    ds = open_glofas_file(data_path)
                except Exception as e:
                    print(f"    ERROR reading downloaded file {data_path}: {e}")
                    continue

                candidate_vars = [v for v in ds.data_vars if "dis" in v.lower()]
                if not candidate_vars:
                    print(f"    ERROR: no discharge-like variable found in {data_path}. "
                          f"Variables present: {list(ds.data_vars)}")
                    ds.close()
                    continue
                var_name = candidate_vars[0]

                nearest = ds[var_name].sel(latitude=lat, longitude=lon, method="nearest")
                series = nearest.to_dataframe()[var_name]
                for date, value in series.items():
                    gauge_rows.append({
                        "date": pd.Timestamp(date).date(),
                        "gauge_id": gauge_id,
                        "discharge_m3s": float(value) if not np.isnan(value) else np.nan,
                    })
                ds.close()
                year_days_extracted += len(series)

            print(f"    {year}: {year_days_extracted} days extracted "
                  f"(across {len(data_paths)} sub-request(s))")

        all_rows.extend(gauge_rows)
        print(f"  Gauge {gauge_id} total: {len(gauge_rows)} days")

    if not all_rows:
        print("\nNo data extracted -- check errors above.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df):,} rows to {OUT_PATH}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Gauges covered: {sorted(df['gauge_id'].unique())}")
    print(f"Missing values: {df['discharge_m3s'].isna().sum()}")
    print("\nNext step: run 06_chirps_glofas_historical_backfill.py")


if __name__ == "__main__":
    main()