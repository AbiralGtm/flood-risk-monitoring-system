import json
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi import HTTPException

from app.config import DATA_PATH, RIVER_POLYGON_PATH, WARD_BOUNDARY_PATH


def _reproject_to_wgs84(data: dict[str, Any]) -> dict[str, Any]:
    """Reproject a GeoJSON FeatureCollection to EPSG:4326 (WGS84) if it
    carries a different CRS.

    Leaflet (and the GeoJSON spec's default) expect lon/lat in WGS84. The
    source file here is in EPSG:32645 (UTM Zone 45N, meters) -- serving it
    as-is would put UTM meter coordinates where the frontend expects
    degrees, and the map would fail to render / misplace everything.
    """
    crs_member = data.get("crs")
    if not crs_member:
        # No crs member present: per the GeoJSON spec, that means WGS84 already.
        return data

    crs_name = crs_member.get("properties", {}).get("name")
    if not crs_name or "4326" in crs_name:
        return data

    import geopandas as gpd

    gdf = gpd.GeoDataFrame.from_features(data["features"], crs=crs_name)
    reprojected = json.loads(gdf.to_crs(4326).to_json())
    reprojected.pop("crs", None)
    return reprojected


def _read_geojson_file(path: Path) -> dict[str, Any]:
    """Read, validate, and reproject a GeoJSON FeatureCollection from disk."""
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Dataset unavailable",
                "message": f"GeoJSON file not found at {path}",
            },
        )

    try:
        with path.open("r", encoding="utf-8") as geojson_file:
            data = json.load(geojson_file)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Invalid dataset",
                "message": "GeoJSON file could not be parsed",
            },
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Dataset read failed",
                "message": "GeoJSON file could not be read",
            },
        ) from exc

    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Invalid dataset structure",
                "message": "Expected a GeoJSON FeatureCollection with a features list",
            },
        )

    return _reproject_to_wgs84(data)


@lru_cache(maxsize=1)
def get_geojson_data() -> dict[str, Any]:
    """Cache the grid GeoJSON in memory for repeated API access."""
    return _read_geojson_file(DATA_PATH)


def _clean_river_name(raw_name: str | None) -> str:
    """The source file's Name property is QGIS/KML export junk
    ("narayani polygon", "east-rapti-polygon") -- normalize it to match the
    "Narayani"/"Rapti" convention already used in nearest_major_river.
    """
    key = (raw_name or "").lower()
    if "narayani" in key:
        return "Narayani"
    if "rapti" in key:
        return "Rapti"
    return raw_name or "Unknown"


@lru_cache(maxsize=1)
def get_river_polygons() -> dict[str, Any]:
    """Cache the Narayani/Rapti river corridor polygons (WGS84, cleaned properties)."""
    raw = _read_geojson_file(RIVER_POLYGON_PATH)

    cleaned_features = [
        {
            "type": "Feature",
            "properties": {"river_name": _clean_river_name(feature.get("properties", {}).get("Name"))},
            "geometry": feature.get("geometry"),
        }
        for feature in raw["features"]
    ]

    return build_feature_collection(cleaned_features)


@lru_cache(maxsize=1)
def get_ward_boundaries() -> dict[str, Any]:
    """Cache Bharatpur Metropolitan City's 29 ward boundary polygons (WGS84,
    cleaned properties). See app/config.py's WARD_BOUNDARY_PATH comment for
    the source file's coordinate-system caveat.
    """
    raw = _read_geojson_file(WARD_BOUNDARY_PATH)

    cleaned_features = [
        {
            "type": "Feature",
            "properties": {
                "ward_number": feature.get("properties", {}).get("NEW_WARD_N"),
                "municipality": feature.get("properties", {}).get("GaPa_NaPa"),
                "district": feature.get("properties", {}).get("DISTRICT"),
                "area_sqkm": feature.get("properties", {}).get("Area_SQKM"),
            },
            "geometry": feature.get("geometry"),
        }
        for feature in raw["features"]
    ]

    return build_feature_collection(cleaned_features)


@lru_cache(maxsize=1)
def get_grid_ward_mapping() -> dict[str, int | None]:
    """Maps each grid_id to the ward_number (1-29) of its nearest ward
    polygon. Computed once and cached -- both grid and ward geometries are
    static, this never changes at runtime.

    Uses nearest-neighbor rather than strict point-in-polygon: the grid
    dataset and the ward boundary file (see app/config.py's
    WARD_BOUNDARY_PATH comment on its empirically-calibrated coordinate
    system) come from independent sources with slightly different
    digitizing, so ~5% of grid centroids don't land strictly inside any
    ward polygon at their shared seams (max observed gap ~222m, under one
    250m grid cell width). Falling back to nearest-ward assignment for
    those means every scoreable grid gets a ward, instead of ~5% silently
    dropping out of any ward-level report.

    Grid coverage is filtered to the flood-relevant study area near the
    rivers (see app/config.py's DATA_PATH comment), not the full municipal
    footprint -- several wards far from the rivers legitimately end up with
    zero grids here. That's expected, not a bug this mapping can fix.
    """
    import geopandas as gpd

    grid_gdf = gpd.GeoDataFrame.from_features(get_geojson_data()["features"], crs="EPSG:4326")
    ward_gdf = gpd.GeoDataFrame.from_features(get_ward_boundaries()["features"], crs="EPSG:4326")

    # Reproject to a metric CRS (matches the grid data's own source CRS) for
    # an accurate centroid/nearest-neighbor computation -- doing this in
    # degrees would distort distances.
    grid_utm = grid_gdf.to_crs(32645)
    ward_utm = ward_gdf.to_crs(32645)

    centroids = grid_utm.copy()
    centroids["geometry"] = centroids.geometry.centroid

    joined = gpd.sjoin_nearest(
        centroids, ward_utm[["ward_number", "geometry"]], how="left", distance_col="_ward_dist_m"
    )
    joined = joined.drop_duplicates(subset="grid_id")

    mapping: dict[str, int | None] = {}
    for _, row in joined.iterrows():
        grid_id = str(row["grid_id"])
        ward_number = row["ward_number"]
        mapping[grid_id] = int(ward_number) if ward_number == ward_number else None  # NaN check
    return mapping


def build_feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a valid GeoJSON FeatureCollection object."""
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def get_feature_by_grid_id(grid_id: str) -> dict[str, Any]:
    """Return one feature matching the given grid_id."""
    for feature in get_geojson_data()["features"]:
        properties = feature.get("properties", {})
        if str(properties.get("grid_id")) == str(grid_id):
            return feature

    raise HTTPException(
        status_code=404,
        detail={
            "error": "Grid not found",
            "message": f"No grid found for grid_id '{grid_id}'",
        },
    )


def unique_property_values(property_name: str) -> list[str]:
    """Return sorted unique non-empty values for a property."""
    values = {
        str(feature.get("properties", {}).get(property_name)).strip()
        for feature in get_geojson_data()["features"]
        if feature.get("properties", {}).get(property_name) not in (None, "")
    }
    return sorted(values)


def count_property_values(property_name: str) -> dict[str, int]:
    """Count occurrences of a property value across all features."""
    counts: dict[str, int] = {}
    for feature in get_geojson_data()["features"]:
        value = feature.get("properties", {}).get(property_name, "Unknown")
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def numeric_property_values(property_name: str) -> list[float]:
    """Extract numeric values for summary calculations."""
    values: list[float] = []
    for feature in get_geojson_data()["features"]:
        raw_value = feature.get("properties", {}).get(property_name)
        if isinstance(raw_value, (int, float)):
            values.append(float(raw_value))
    return values


def safe_mean(values: list[float]) -> float | None:
    """Return the average when values exist."""
    if not values:
        return None
    return round(mean(values), 3)


def safe_min(values: list[float]) -> float | None:
    """Return the minimum when values exist."""
    if not values:
        return None
    return round(min(values), 3)


def safe_max(values: list[float]) -> float | None:
    """Return the maximum when values exist."""
    if not values:
        return None
    return round(max(values), 3)


def matches_filters(
    feature: dict[str, Any],
    risk_class: str | None,
    lulc: str | None,
    max_distance: float | None,
    min_elevation: float | None,
    max_elevation: float | None,
    static_class_by_grid: dict[str, str] | None = None,
) -> bool:
    """Check whether a feature matches the provided filter criteria.

    static_class_by_grid: precomputed {grid_id: static_susceptibility_class}
    map. Pass this in (computed once) rather than recomputing per feature
    when filtering a whole collection by risk_class.
    """
    properties = feature.get("properties", {})

    if risk_class:
        grid_class = (static_class_by_grid or {}).get(properties.get("grid_id"))
        if str(grid_class or "").lower() != risk_class.lower():
            return False

    if lulc and str(properties.get("lulc_dominant", "")).lower() != lulc.lower():
        return False

    distance = properties.get("nearest_river_distance")
    if max_distance is not None:
        if not isinstance(distance, (int, float)) or float(distance) > max_distance:
            return False

    elevation = properties.get("elevation_mean")
    if min_elevation is not None:
        if not isinstance(elevation, (int, float)) or float(elevation) < min_elevation:
            return False

    if max_elevation is not None:
        if not isinstance(elevation, (int, float)) or float(elevation) > max_elevation:
            return False

    return True

