from fastapi import APIRouter

from app.ml.static_risk import static_risk_class_counts
from app.utils import (
    count_property_values,
    get_geojson_data,
    numeric_property_values,
    safe_max,
    safe_mean,
    safe_min,
)

router = APIRouter()


@router.get("/summary")
def get_summary() -> dict[str, object]:
    """Return summary statistics for the full grid dataset."""
    elevation_values = numeric_property_values("elevation_mean")
    slope_values = numeric_property_values("slope_mean")
    distance_values = numeric_property_values("nearest_river_distance")

    return {
        "total_grids": len(get_geojson_data()["features"]),
        "static_risk_class_counts": static_risk_class_counts(),
        "lulc_dominant_counts": count_property_values("lulc_dominant"),
        "average_elevation": safe_mean(elevation_values),
        "average_slope": safe_mean(slope_values),
        "average_distance_to_river": safe_mean(distance_values),
        "min_elevation": safe_min(elevation_values),
        "max_elevation": safe_max(elevation_values),
        "min_slope": safe_min(slope_values),
        "max_slope": safe_max(slope_values),
    }

