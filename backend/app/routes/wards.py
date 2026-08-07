from typing import Any

from fastapi import APIRouter, Depends, Query

from app.ml.model_loader import BEST_MODEL_ID
from app.routes.risk_map import condition_params, score_all_grids
from app.utils import get_grid_ward_mapping, get_ward_boundaries

router = APIRouter()

# Severity weights for ranking wards by risk -- higher predicted risk classes
# count for more. This is a simple, transparent scale (0-3), not a
# probability; risk_score is the mean weight across a ward's scored grids.
_RISK_WEIGHT = {"Low": 0, "Medium": 1, "High": 2, "Danger": 3}
_RISK_CLASSES = ("Low", "Medium", "High", "Danger")


def _ward_metadata_by_number() -> dict[int, dict[str, Any]]:
    return {
        feature["properties"]["ward_number"]: feature["properties"]
        for feature in get_ward_boundaries()["features"]
    }


@router.get("/wards/risk-report")
def get_ward_risk_report(
    model: str = Query(default=BEST_MODEL_ID),
    conditions: dict[str, Any] = Depends(condition_params),
) -> dict[str, Any]:
    """Aggregate the risk map by ward and rank wards by risk, for the given
    model + conditions -- same inputs as GET /api/risk-map, since ward-level
    risk is just as condition-dependent as grid-level risk (there's no
    single static "riskiest ward").

    Each grid is assigned to its nearest ward (see get_grid_ward_mapping()
    in app/utils.py for why "nearest" rather than strict point-in-polygon).
    `risk_score` is the mean severity weight (Low=0, Medium=1, High=2,
    Danger=3) across a ward's scored grids -- a simple, transparent ranking
    signal, not a probability. Wards are sorted by risk_score descending,
    tie-broken by percentage of High/Danger grids, then raw Danger count.

    `wards_with_no_grid_coverage` lists wards with zero grids in the
    model's study-area population (it's filtered to the flood-relevant area
    near the rivers, not the full municipal footprint -- see app/config.py's
    DATA_PATH comment) -- these wards simply can't be risk-ranked from this
    dataset, not necessarily low-risk.
    """
    enriched_features, skipped = score_all_grids(model, conditions)
    mapping = get_grid_ward_mapping()
    ward_meta = _ward_metadata_by_number()

    per_ward: dict[int, dict[str, Any]] = {}
    unassigned_grid_count = 0
    for feature in enriched_features:
        properties = feature["properties"]
        ward_number = mapping.get(str(properties.get("grid_id")))
        if ward_number is None:
            unassigned_grid_count += 1
            continue

        bucket = per_ward.setdefault(
            ward_number, {cls: 0 for cls in _RISK_CLASSES} | {"weight_sum": 0}
        )
        risk_class = properties["predicted_risk_class"]
        bucket[risk_class] = bucket.get(risk_class, 0) + 1
        bucket["weight_sum"] += _RISK_WEIGHT.get(risk_class, 0)

    wards: list[dict[str, Any]] = []
    for ward_number, bucket in per_ward.items():
        counts = {cls: bucket[cls] for cls in _RISK_CLASSES}
        total = sum(counts.values())
        high_danger_count = counts["High"] + counts["Danger"]
        meta = ward_meta.get(ward_number, {})
        wards.append(
            {
                "ward_number": ward_number,
                "municipality": meta.get("municipality"),
                "district": meta.get("district"),
                "area_sqkm": meta.get("area_sqkm"),
                "grid_count": total,
                "risk_class_counts": counts,
                "high_danger_count": high_danger_count,
                "high_danger_pct": round(high_danger_count / total * 100, 2) if total else 0.0,
                "risk_score": round(bucket["weight_sum"] / total, 4) if total else 0.0,
                "dominant_risk_class": max(counts, key=counts.get) if total else None,
            }
        )

    wards.sort(
        key=lambda w: (-w["risk_score"], -w["high_danger_pct"], -w["risk_class_counts"]["Danger"])
    )
    for rank, ward in enumerate(wards, start=1):
        ward["rank"] = rank

    covered_wards = {w["ward_number"] for w in wards}
    no_coverage = sorted(set(ward_meta) - covered_wards)

    return {
        "model_id": model,
        "conditions": conditions,
        "total_grids_scored": len(enriched_features),
        "skipped_grid_count": len(skipped),
        "unassigned_grid_count": unassigned_grid_count,
        "wards": wards,
        "wards_with_no_grid_coverage": no_coverage,
    }
