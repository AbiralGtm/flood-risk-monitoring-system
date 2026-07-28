from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import ALLOWED_ORIGINS
from app.db.session import init_db
from app.routes.conditions import router as conditions_router
from app.routes.grids import router as grids_router
from app.routes.history import router as history_router
from app.routes.layers import router as layers_router
from app.routes.predict import router as predict_router
from app.routes.risk import router as risk_router
from app.routes.risk_map import router as risk_map_router
from app.routes.summary import router as summary_router
from app.routes.wards import router as wards_router

app = FastAPI(
    title="flood-risk-api",
    version="1.0.0",
    description="FastAPI backend for Bharatpur flood-risk GIS dashboard data.",
)


@app.on_event("startup")
def _create_db_tables() -> None:
    """Create history tables (rainfall/river-level/forecast/prediction logs)
    if they don't exist yet. See app/db/session.py -- no migrations tool
    wired in, this only ever adds missing tables, never alters existing ones.
    """
    init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Return JSON for unexpected runtime errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
        },
    )


@app.get("/")
def root() -> dict[str, str]:
    """Simple welcome endpoint for quick API checks."""
    return {"message": "Welcome to the flood-risk-api service."}


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight health check endpoint."""
    return {"status": "ok", "service": "flood-risk-api"}


app.include_router(grids_router, prefix="/api", tags=["grids"])
app.include_router(risk_router, prefix="/api", tags=["risk"])
app.include_router(summary_router, prefix="/api", tags=["summary"])
app.include_router(predict_router, prefix="/api", tags=["predict"])
app.include_router(risk_map_router, prefix="/api", tags=["risk-map"])
app.include_router(layers_router, prefix="/api", tags=["layers"])
app.include_router(conditions_router, prefix="/api", tags=["conditions"])
app.include_router(history_router, prefix="/api", tags=["history"])
app.include_router(wards_router, prefix="/api", tags=["wards"])

