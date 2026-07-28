from typing import Any

from fastapi import APIRouter, Query

from app.ml.static_risk import get_static_risk_scores
from app.utils import (
    build_feature_collection,
    get_feature_by_grid_id,
    get_geojson_data,
    matches_filters,
    unique_property_values,
)

router = APIRouter()


@router.get("/grids")
def get_grids() -> dict[str, Any]:
    """Return the full grid GeoJSON FeatureCollection."""
    return get_geojson_data()


@router.get("/filters/lulc")
def get_lulc_filters() -> dict[str, list[str]]:
    """Return unique dominant LULC values for frontend filtering."""
    return {"lulc_dominant_values": unique_property_values("lulc_dominant")}


@router.get("/filters/risk-classes")
def get_risk_class_filters() -> dict[str, list[str]]:
    """Return the static susceptibility risk classes available for filtering."""
    classes = sorted({entry["static_susceptibility_class"] for entry in get_static_risk_scores()})
    return {"risk_classes": classes}


@router.get("/grids/filter")
def filter_grids(
    risk_class: str | None = Query(default=None),
    lulc: str | None = Query(default=None),
    max_distance: float | None = Query(default=None, ge=0),
    min_elevation: float | None = Query(default=None),
    max_elevation: float | None = Query(default=None),
) -> dict[str, Any]:
    """Filter grid features using simple property-based query parameters."""
    features = get_geojson_data()["features"]
    static_class_by_grid = (
        {entry["grid_id"]: entry["static_susceptibility_class"] for entry in get_static_risk_scores()}
        if risk_class
        else None
    )
    filtered_features = [
        feature
        for feature in features
        if matches_filters(
            feature=feature,
            risk_class=risk_class,
            lulc=lulc,
            max_distance=max_distance,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            static_class_by_grid=static_class_by_grid,
        )
    ]
    return build_feature_collection(filtered_features)


@router.get("/grids/{grid_id}")
def get_grid(grid_id: str) -> dict[str, Any]:
    """Return one grid feature by its grid_id."""
    return get_feature_by_grid_id(grid_id)
