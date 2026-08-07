from typing import Any

from fastapi import APIRouter

from app.utils import get_river_polygons, get_ward_boundaries

router = APIRouter()


@router.get("/layers/rivers")
def get_rivers() -> dict[str, Any]:
    """Return the Narayani + East Rapti river corridor polygons (WGS84) as a
    reference layer for the map. Each feature's properties has one field,
    river_name ("Narayani" | "Rapti")."""
    return get_river_polygons()


@router.get("/layers/wards")
def get_wards() -> dict[str, Any]:
    """Return Bharatpur Metropolitan City's 29 administrative ward boundary
    polygons (WGS84) as a reference layer for the map. Each feature's
    properties has ward_number (1-29), municipality, district, and
    area_sqkm. See app/config.py's WARD_BOUNDARY_PATH comment for a note on
    the source coordinate system."""
    return get_ward_boundaries()
