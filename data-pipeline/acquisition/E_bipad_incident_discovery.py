"""
Fetch flood incident records from BIPAD's /incident/ endpoint for
Bharatpur (municipality=35001), hazard=11 (likely flood -- verify from
the actual data, e.g. check a hazardName/hazard_type field in the
response). This endpoint has no 'fields=' whitelist in the URL you found,
so it should return the full default record -- but given we've hit two
field-naming surprises already (rain: camelCase output vs snake_case
filter; river: same pattern reversed), print the raw structure FIRST
before assuming any key names.

Run this, then share the printed output -- once we see the real field
names (location/point format, date field name, description field,
source field), I'll write the parser that converts these into your
flood_evidence_bharatpur.csv schema.
"""

import requests
import json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

BASE_URL = "http://bipadportal.gov.np/api/v1/incident/"

params = {
    "hazard": 11,
    "data_source": "drr_api",
    "incident_on__gt": "2021-05-01T00:00:00+05:45",
    "incident_on__lt": "2026-07-25T23:59:59+05:45",
    "municipality": 35001,
}

r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
print("Status code:", r.status_code)
print("URL called:", r.url)
print()

data = r.json()
print("Top-level keys:", list(data.keys()) if isinstance(data, dict) else "list")

results = data.get("results", data if isinstance(data, list) else [])
count = data.get("count") if isinstance(data, dict) else len(results)
print(f"Count field: {count}")
print(f"Records in this page: {len(results)}")
print(f"Next page URL: {data.get('next') if isinstance(data, dict) else 'n/a'}")

if results:
    print("\n--- First record, full raw structure ---")
    print(json.dumps(results[0], indent=2))
    if len(results) > 1:
        print("\n--- Second record (to compare field consistency) ---")
        print(json.dumps(results[1], indent=2))
else:
    print("\nNo results for hazard=11 -- this code may not mean 'flood'.")
    print("Try removing the hazard filter to see what hazard values/names exist:")
    print("  params without 'hazard' key, then inspect a hazard-related field")
    print("  in the returned records to find the right code.")
