"""
STEP 0 (one-off) -- fetch a small sample from the confirmed BIPAD rain
endpoint and dump it to a file so we can see the EXACT field structure
(especially what's inside "averages") before writing the bulk downloader.

Run this, then share the printed output (or the saved JSON file) back --
that tells us exactly how to parse `rainfall_1h_mm` etc. out of the
response correctly on the first try.
"""

import requests
import json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

BASE_URL = "http://bipadportal.gov.np/api/v1/rain/"

# Small, recent window -- just enough to see real records without a huge payload.
params = {
    "title": "Rampur_AWS",
    "is_hourly": "true",
    "measured_on__gt": "2026-07-12T00:00:00+05:45",
    "measured_on__lt": "2026-07-13T00:00:00+05:45",
    "fields": "id,created_on,measured_on,title,basin,point,averages,status,station",
    "limit": -1,
}

r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=20)
print("Status code:", r.status_code)
print("URL actually called:", r.url)
print()

try:
    data = r.json()
except Exception as e:
    print("Still not JSON. First 1000 chars of raw response:")
    print(r.text[:1000])
    raise SystemExit

# Save full response for inspection
with open("bipad_sample_response.json", "w") as f:
    json.dump(data, f, indent=2)
print("Saved full response to bipad_sample_response.json")
print()

# Print a compact view of the structure
if isinstance(data, dict) and "results" in data:
    records = data["results"]
    print(f"Response is paginated. Total count field: {data.get('count')}")
elif isinstance(data, list):
    records = data
else:
    records = [data]

print(f"Number of records in this window: {len(records)}")
if records:
    print("\n--- First record, full structure ---")
    print(json.dumps(records[0], indent=2))
    print("\n--- Just the 'averages' field of record 0 (this is what we need to parse) ---")
    print(json.dumps(records[0].get("averages"), indent=2))
