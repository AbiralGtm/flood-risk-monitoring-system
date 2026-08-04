# main.py (FastAPI)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
app = FastAPI()

# Load once at startup, not per-request
model_bundle = joblib.load("model_xgboost.joblib")
model = model_bundle["model"]
FEATURES = model_bundle["features"]
label_encoder = model_bundle.get("label_encoder")

# Load static grid features once at startup too
static_df = pd.read_csv("ml_training_dataset.csv").drop_duplicates(subset="grid_id")
static_lookup = static_df.set_index("grid_id")

class PredictRequest(BaseModel):
    grid_id: str
    rainfall_1h_mm: float
    rainfall_3h_mm: float
    rainfall_6h_mm: float
    rainfall_12h_mm: float
    rainfall_24h_mm: float
    water_level_m: float | None = None

def rainfall_score(r: PredictRequest) -> float:
    return min(1.0, max(
        r.rainfall_1h_mm/60, r.rainfall_3h_mm/80, r.rainfall_6h_mm/100,
        r.rainfall_12h_mm/120, r.rainfall_24h_mm/140,
    ))

def river_level_score(water_level, warning_level, danger_level) -> float:
    if water_level is None or pd.isna(warning_level) or pd.isna(danger_level):
        return 0.3
    if water_level >= danger_level:
        return 1.0
    if water_level >= warning_level:
        return 0.5 + 0.5*(water_level-warning_level)/(danger_level-warning_level)
    return max(0.0, 0.5*water_level/warning_level)

@app.post("/predict")
def predict(req: PredictRequest):
    if req.grid_id not in static_lookup.index:
        raise HTTPException(404, f"grid_id {req.grid_id} not found")

    grid_static = static_lookup.loc[req.grid_id]
    feat = {f: grid_static.get(f, 0) for f in FEATURES}

    feat["rainfall_1h_mm"] = req.rainfall_1h_mm
    feat["rainfall_3h_mm"] = req.rainfall_3h_mm
    feat["rainfall_6h_mm"] = req.rainfall_6h_mm
    feat["rainfall_12h_mm"] = req.rainfall_12h_mm
    feat["rainfall_24h_mm"] = req.rainfall_24h_mm
    feat["rainfall_score"] = rainfall_score(req)
    feat["water_level_m"] = req.water_level_m if req.water_level_m is not None else np.nan
    feat["river_level_score"] = river_level_score(
        req.water_level_m, grid_static.get("warning_level_m"), grid_static.get("danger_level_m")
    )

    X = pd.DataFrame([feat])[FEATURES]
    pred_raw = model.predict(X)[0]
    pred = label_encoder.inverse_transform([pred_raw])[0] if label_encoder is not None else pred_raw
    confidence = float(model.predict_proba(X).max())

    return {
        "grid_id": req.grid_id,
        "predicted_risk_class": pred,
        "confidence": round(confidence, 3),
        "nearest_major_river": grid_static.get("nearest_major_river"),
        "nearest_river_distance_m": round(float(grid_static.get("nearest_river_distance", 0)), 1),
    }