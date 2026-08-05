"""Loads and caches the trained flood-risk classifiers.

Each .joblib file was saved as a dict with at minimum:
    {"model": <estimator>, "features": [<column names in training order>]}
LogReg additionally stores "scaler" (StandardScaler applied before predict).
XGBoost variants additionally store "label_encoder" (maps predicted ints
back to class name strings).

All four models were trained on the same 14-column feature set (see
app/ml/features.py's module docstring for the 2026-08 retrain that dropped
four sub-hourly rainfall fields and raw water level from 18 down to 14) and
expose classes_ in the same sorted order: ['Danger', 'High', 'Low', 'Medium'].

xgboost_tuned (added 2026-08-08) additionally stores a custom decision
threshold for the Danger class:
    {"danger_threshold": float, "threshold_method": str,
     "min_recall_floor": float, "threshold_derived_from_n_rows": int,
     "threshold_derived_from_split": str,
     "held_out_test_danger_precision": float, "held_out_test_danger_recall": float}
Despite the "recall_floor" name, danger_threshold (0.5936) is a GATE, not a
force-to-Danger override: a grid is only classified Danger if the model's own
P(Danger) clears this threshold; otherwise the prediction falls back to the
best of the *other three* classes, even if Danger was plain argmax's winner
at a lower probability. (threshold_method =
"high_recall_floor_0.95_derived_from_val2025" describes how the threshold was
CHOSEN -- the highest cutoff that still kept recall >= a 0.95 floor on a 2025
validation split -- not the rule applied at inference time.) Concretely:
since danger_threshold > 0.5, and this is a single-label 4-class softmax, if
P(Danger) is already >= 0.5 no other class can have higher probability
(remaining mass < 0.5), so this threshold can only ever remove Danger
classifications that plain argmax would have made at 0.30-0.59 confidence,
never add ones argmax wouldn't already make. That is: xgboost_tuned trades
some Danger recall for higher Danger precision versus plain "xgboost" --
confirmed by app/config.py's MODEL_METRICS (fewer, more confident Danger
calls, same underlying model weights as plain "xgboost", only the decision
rule differs). Ignoring this field and doing plain argmax (like the other
three models) would make the tuning a no-op at serving time, so
ModelBundle.predict_batch() below applies it explicitly for any bundle that
has one. See _evaluate_one_model() in app/ml/evaluate.py for the matching
evaluation-time logic -- the metrics recorded in app/config.py's
MODEL_METRICS for this model reflect the gate being applied, not plain
argmax, so they describe what this model actually returns in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import MODELS_DIR, MODEL_METRICS

# Registry of available models. "xgboost_tuned" is the best model (2026-08-08:
# a recall-floor-thresholded version of the xgboost model below -- see the
# module docstring -- 0.9401 held-out Danger recall vs. plain xgboost's
# 0.959 test-set Danger recall on a differently-defined evaluation; see
# app/config.py's MODEL_METRICS, reproducible via `python -m app.ml.evaluate`,
# for the apples-to-apples comparison against the same fixed test split used
# for all four models) and is the default used by /api/predict. The other
# three are exposed for side-by-side comparison only.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "xgboost_tuned": {
        # Danger-recall-floor-thresholded version of "xgboost" below (same
        # underlying training run, mtime 2026-08-08 -- see module docstring
        # for the danger_threshold field this bundle carries). This is now
        # the default/best model.
        "file": "model_xgboost_tuned.joblib",
        "display_name": "XGBoost (Tuned)",
        "algorithm": "XGBClassifier",
    },
    "xgboost": {
        # Plain-argmax XGBoost, no custom decision threshold (mtime
        # 2026-08-04 20:48). Kept as its own selectable model for comparison
        # against the thresholded xgboost_tuned above -- not deprecated, just
        # no longer the default. A pre-fine-tuning version is also kept at
        # models/model_xgboost_v1.joblib for reference but is NOT loaded
        # anywhere -- confirmed via a full-repo grep, zero references outside
        # this comment and the dataset CSV's own filename column.
        "file": "model_xgboost.joblib",
        "display_name": "XGBoost",
        "algorithm": "XGBClassifier",
    },
    "random_forest": {
        "file": "model_random_forest.joblib",
        "display_name": "Random Forest",
        "algorithm": "RandomForestClassifier",
    },
    "logreg": {
        "file": "model_logreg.joblib",
        "display_name": "Logistic Regression",
        "algorithm": "LogisticRegression",
    },
}

BEST_MODEL_ID = "xgboost_tuned"


def _predict_proba_safe(model: Any, X: Any) -> Any:
    """model.predict_proba(X), with a fallback for a known scikit-learn
    version-skew bug: the pickled LogisticRegression model raises
    `AttributeError: 'LogisticRegression' object has no attribute
    'multi_class'` under a scikit-learn install older than the one it was
    trained/pickled with (see requirements.txt's comment on this exact
    issue -- this repo pins 1.9.0, but a dev/CI environment can still end up
    with an older one installed). app/ml/evaluate.py already carries this
    same workaround for its own offline scoring path
    (`_logreg_predict_proba`); this is the equivalent for live serving --
    without it, POST /api/predict/compare silently returns an `error` entry
    for logreg instead of a real comparison, which defeats the point of a
    4-model side-by-side comparison. Manual softmax over the same linear
    decision function predict_proba would use internally; verified
    numerically identical to a working predict_proba on a test row.
    """
    try:
        return model.predict_proba(X)
    except AttributeError:
        X_arr = np.asarray(X, dtype=float)  # X may be a DataFrame here (see predict_batch); the
        # matrix ops below need a plain ndarray, matching evaluate.py's equivalent workaround.
        z = X_arr @ model.coef_.T + model.intercept_  # == model.decision_function(X)
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)


@dataclass
class ModelBundle:
    model_id: str
    display_name: str
    algorithm: str
    model: Any
    features: list[str]
    scaler: Any | None = None
    label_encoder: Any | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    # Optional recall-floor decision threshold for the Danger class -- see
    # this module's docstring. None for every model except xgboost_tuned.
    danger_threshold: float | None = None

    @property
    def classes(self) -> list[str]:
        """Class labels in the exact order predict_proba columns are returned."""
        if self.label_encoder is not None:
            return list(self.label_encoder.classes_)
        return list(self.model.classes_)

    def predict(self, feature_row: dict[str, float]) -> dict[str, Any]:
        """Run one prediction. feature_row must contain every key in self.features."""
        return self.predict_batch([feature_row])[0]

    def predict_batch(self, feature_rows: list[dict[str, float]]) -> list[dict[str, Any]]:
        """Run predictions for many rows in a single vectorized call."""
        if not feature_rows:
            return []

        for i, row in enumerate(feature_rows):
            missing = [f for f in self.features if f not in row]
            if missing:
                raise ValueError(
                    f"Missing features for model '{self.model_id}' at row {i}: {missing}"
                )

        frame = pd.DataFrame(
            [[row[f] for f in self.features] for row in feature_rows], columns=self.features
        )

        if self.scaler is not None:
            frame = pd.DataFrame(self.scaler.transform(frame), columns=self.features)

        proba_matrix = _predict_proba_safe(self.model, frame)
        class_names = self.classes
        danger_idx = class_names.index("Danger") if "Danger" in class_names else None

        results: list[dict[str, Any]] = []
        for proba in proba_matrix:
            probabilities = {cls: round(float(p), 6) for cls, p in zip(class_names, proba)}

            # Danger-class gate: if this bundle carries a danger_threshold,
            # only classify Danger when P(Danger) clears it; otherwise fall
            # back to the best of the *other* classes even if Danger was
            # plain argmax's winner at a lower probability. This raises the
            # bar for Danger (higher precision, some recall given up) rather
            # than lowering it -- see module docstring for why a threshold
            # above 0.5 can only remove Danger calls, never add ones plain
            # argmax wouldn't already make. Skipping this check would
            # silently make xgboost_tuned's whole point a no-op.
            if self.danger_threshold is not None and danger_idx is not None:
                if proba[danger_idx] >= self.danger_threshold:
                    predicted_class = "Danger"
                else:
                    other_idx = [i for i in range(len(class_names)) if i != danger_idx]
                    best_other = max(other_idx, key=lambda i: proba[i])
                    predicted_class = class_names[best_other]
            else:
                predicted_idx = int(proba.argmax())
                predicted_class = class_names[predicted_idx]

            results.append(
                {
                    "model_id": self.model_id,
                    "predicted_risk_class": predicted_class,
                    "risk_probability": probabilities[predicted_class],
                    "probabilities": probabilities,
                }
            )
        return results


def _load_bundle(model_id: str) -> ModelBundle:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(MODEL_REGISTRY)}")

    entry = MODEL_REGISTRY[model_id]
    path: Path = MODELS_DIR / entry["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Place the .joblib files in {MODELS_DIR}."
        )

    payload = joblib.load(path)

    return ModelBundle(
        model_id=model_id,
        display_name=entry["display_name"],
        algorithm=entry["algorithm"],
        model=payload["model"],
        features=list(payload["features"]),
        scaler=payload.get("scaler"),
        label_encoder=payload.get("label_encoder"),
        metrics=MODEL_METRICS.get(model_id, {}),
        danger_threshold=payload.get("danger_threshold"),
    )


@lru_cache(maxsize=len(MODEL_REGISTRY))
def get_model(model_id: str) -> ModelBundle:
    """Load (once) and cache a model bundle by id."""
    return _load_bundle(model_id)


def list_model_ids() -> list[str]:
    return list(MODEL_REGISTRY)


def get_all_models() -> dict[str, ModelBundle]:
    return {model_id: get_model(model_id) for model_id in MODEL_REGISTRY}
