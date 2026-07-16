import requests
import pandas as pd
import time

base_url = "https://bipadportal.gov.np/api/v1/rain/"

params = {
    "district": "",
    "is_daily": "unknown",
    "is_hourly": "true",
    "limit": 1000,
    "measured_on__gt": "2020-01-01T00:00:00+05:45",
    "measured_on__lt": "2026-07-01T23:59:59+05:45",
    "municipality": "",
    "ordering": "created_on",
    "province": "",
    "station": "",
    "station_series_id": "",
    "status__iexact": "",
    "title": "Narayani Field Office",
    "ward": "",
    "format": "json"
}

all_rows = []
url = base_url
page = 1

while url:
    print(f"Downloading page {page}...")

    response = requests.get(url, params=params if page == 1 else None, timeout=30)
    response.raise_for_status()

    data = response.json()
    results = data.get("results", [])

    if not results:
        break

    for item in results:
        row = {
            "id": item.get("id"),
            "title": item.get("title"),
            "measured_on": item.get("measuredOn"),
            "status": item.get("status"),
            "station_series_id": item.get("stationSeriesId"),
            "station": item.get("station"),
            "ward": item.get("ward"),
            "municipality": item.get("municipality"),
            "district": item.get("district"),
            "province": item.get("province"),
            "longitude": None,
            "latitude": None,
            "rainfall_1h_mm": None,
            "rainfall_3h_mm": None,
            "rainfall_6h_mm": None,
            "rainfall_12h_mm": None,
            "rainfall_24h_mm": None,
        }

        point = item.get("point")
        if point and point.get("coordinates"):
            row["longitude"] = point["coordinates"][0]
            row["latitude"] = point["coordinates"][1]

        for avg in item.get("averages", []):
            interval = avg.get("interval")
            value = avg.get("value")

            if interval == 1:
                row["rainfall_1h_mm"] = value
            elif interval == 3:
                row["rainfall_3h_mm"] = value
            elif interval == 6:
                row["rainfall_6h_mm"] = value
            elif interval == 12:
                row["rainfall_12h_mm"] = value
            elif interval == 24:
                row["rainfall_24h_mm"] = value

        all_rows.append(row)

    url = data.get("next")
    params = None
    page += 1

    time.sleep(0.3)

df = pd.DataFrame(all_rows)

print("Total downloaded records:", len(df))
print("Date range:")
print(df["measured_on"].min(), "to", df["measured_on"].max())

df.to_csv("narayani_field_office_hourly_rainfall_2020_2026.csv", index=False)

print("Saved CSV successfully.")