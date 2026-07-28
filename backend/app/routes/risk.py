from typing import Any

from fastapi import APIRouter

from app.ml.static_risk import compute_static_susceptibility_score, get_static_risk_scores
from app.utils import build_feature_collection, get_geojson_data

router = APIRouter()


@router.get("/risk/static")
def get_static_risk() -> dict[str, Any]:
    """Return features including static-only susceptibility fields for map display.

    NOTE: this reflects static susceptibility only (terrain, river distance,
    land cover). It is not the trained model's predicted_risk_class, which
    also needs live rainfall/river conditions -- see POST /api/predict.
    """
    classes_by_grid = {entry["grid_id"]: entry["static_susceptibility_class"] for entry in get_static_risk_scores()}

    features_with_risk: list[dict[str, Any]] = []
    for feature in get_geojson_data()["features"]:
        properties = dict(feature.get("properties", {}))
        grid_id = properties.get("grid_id")
        properties["static_susceptibility_score"] = compute_static_susceptibility_score(properties)
        properties["static_susceptibility_class"] = classes_by_grid.get(grid_id)
        features_with_risk.append(
            {
                **feature,
                "properties": properties,
            }
        )

    return build_feature_collection(features_with_risk)
