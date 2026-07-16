"""
Find BIPAD's river/water-level endpoint -- same domain as /rain/, likely
a sibling path. Since we can't browse BIPAD's Network tab ourselves, this
tries several plausible endpoint names and reports which one(s) return
real data, so you don't have to guess/inspect manually if one of these
guesses happens to be right.

If NONE of these work, the fallback is the same manual method that found
/rain/ originally: open BIPAD Portal's river/water-level page in Chrome,
F12 -> Network tab -> filter Fetch/XHR -> look at what URL loads when you
view river level data for a station.
"""

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

BASE = "http://bipadportal.gov.np/api/v1/"

CANDIDATE_ENDPOINTS = [
    "waterlevel/",
    "water-level/",
    "water_level/",
    "river/",
    "riverlevel/",
    "river-level/",
    "river_level/",
    "gauge/",
    "gauge-reading/",
    "hydrology/",
    "discharge/",
]

# Reuse the same params pattern that worked for /rain/, with a known
# station title to test against -- Devghat should exist if this is the
# right endpoint. Try without title filter too, in case the field name
# for station identification differs (e.g. "station_name" instead of "title").
TEST_PARAMS_VARIANTS = [
    {"is_hourly": "true", "limit": 5},
    {"limit": 5},
]

for endpoint in CANDIDATE_ENDPOINTS:
    url = BASE + endpoint
    for params in TEST_PARAMS_VARIANTS:
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            status = r.status_code
            if status == 200:
                try:
                    data = r.json()
                    print(f"*** SUCCESS: {url}  (params={params})")
                    print(f"    Status: {status}, Content-Type: {r.headers.get('Content-Type')}")
                    if isinstance(data, dict):
                        print(f"    Top-level keys: {list(data.keys())}")
                        results = data.get("results", [])
                        if results:
                            print(f"    First result keys: {list(results[0].keys())}")
                            print(f"    First result sample: {results[0]}")
                    print()
                except Exception:
                    print(f"    {url}: status 200 but not valid JSON (likely the JS app shell)")
            else:
                print(f"    {url} (params={params}): status {status}")
        except Exception as e:
            print(f"    {url}: request failed -- {e}")
    print()

print("=" * 70)
print("Review any '*** SUCCESS' lines above. If none appear, the endpoint")
print("name isn't one of these guesses -- fall back to manually inspecting")
print("BIPAD Portal's river-watch page in your browser's Network tab, the")
print("same way /rain/ was originally found.")
