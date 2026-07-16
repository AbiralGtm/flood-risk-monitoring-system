import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from urllib.parse import urlencode

BASE_URL = "https://bipadportal.gov.np/api/v1/incident/"

params = {
    "data_source": "drr_api",
    "district": "",
    "event": "",
    "hazard": 11,  # Flood
    "incident_on__gt": "2020-01-01T00:00:00+05:45",
    "incident_on__lt": "2025-12-30T23:59:59+05:45",
    "incident_type": "",
    "loss__id": "",
    "municipality": 35001,  # Bharatpur Metropolitan City
    "province": "",
    "reported_on__gt": "",
    "reported_on__lt": "",
    "source": "",
    "summary": "",
    "summary_type": "",
    "ward": "",
    "limit": 1000
}

def extract_ward_from_title(title):
    """
    Extract ward number from title like:
    Flood at Bharatpur Metropolitan City-14
    """
    if not title:
        return None

    if "City-" in title:
        try:
            return int(title.split("City-")[-1].split()[0].replace(",", "").strip())
        except:
            return None

    return None


def clean_incident(item):
    point = item.get("point")

    longitude = None
    latitude = None

    if point and point.get("coordinates"):
        longitude = point["coordinates"][0]
        latitude = point["coordinates"][1]

    loss = item.get("loss", {})

    # Sometimes loss is only an ID, sometimes full object
    if not isinstance(loss, dict):
        loss = {}

    row = {
        "incident_id": item.get("id"),
        "title": item.get("title"),
        "title_ne": item.get("titleNe"),
        "incident_date": item.get("incidentOn"),
        "reported_date": item.get("reportedOn"),
        "street_address": item.get("streetAddress"),
        "ward_no_from_title": extract_ward_from_title(item.get("title")),
        "ward_ids": ",".join(map(str, item.get("wards", []))) if item.get("wards") else None,
        "longitude": longitude,
        "latitude": latitude,
        "verified": item.get("verified"),
        "approved": item.get("approved"),
        "source": item.get("source"),
        "data_source": item.get("dataSource"),
        "hazard": item.get("hazard"),

        # Loss fields if available
        "people_death_count": loss.get("peopleDeathCount"),
        "people_missing_count": loss.get("peopleMissingCount"),
        "people_injured_count": loss.get("peopleInjuredCount"),
        "people_affected_count": loss.get("peopleAffectedCount"),
        "family_affected_count": loss.get("familyAffectedCount"),
        "family_relocated_count": loss.get("familyRelocatedCount"),
        "family_evacuated_count": loss.get("familyEvacuatedCount"),
        "houses_destroyed": loss.get("infrastructureDestroyedHouseCount"),
        "houses_affected": loss.get("infrastructureAffectedHouseCount"),
        "estimated_loss": loss.get("estimatedLoss"),
    }

    return row


def fetch_bipad_flood_incidents():
    url = BASE_URL + "?" + urlencode(params)
    print("Fetching:", url)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])
    rows = []

    for item in results:
        row = clean_incident(item)

        # Keep only records with point coordinates
        if row["latitude"] is not None and row["longitude"] is not None:
            rows.append(row)

    return pd.DataFrame(rows)


df = fetch_bipad_flood_incidents()

# Save CSV
csv_path = "historical_flood_evidence_bharatpur_bipad.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"Saved CSV: {csv_path}")
print(df[[
    "incident_id",
    "incident_date",
    "title",
    "street_address",
    "ward_no_from_title",
    "latitude",
    "longitude",
    "verified",
    "approved"
]])

# Save GeoJSON for QGIS
geometry = [
    Point(xy) for xy in zip(df["longitude"], df["latitude"])
]

gdf = gpd.GeoDataFrame(
    df,
    geometry=geometry,
    crs="EPSG:4326"
)

geojson_path = "historical_flood_evidence_bharatpur_bipad.geojson"
gdf.to_file(geojson_path, driver="GeoJSON")

print(f"Saved GeoJSON: {geojson_path}")

# Save UTM GeoPackage for QGIS processing
gdf_utm = gdf.to_crs("EPSG:32645")

gpkg_path = "historical_flood_evidence_bharatpur_bipad_utm.gpkg"
gdf_utm.to_file(
    gpkg_path,
    layer="historical_flood_evidence_bharatpur_bipad_utm",
    driver="GPKG"
)

print(f"Saved UTM GeoPackage: {gpkg_path}")