"""Reproducible model evaluation -- computes the same test_accuracy/macro_f1/
danger_recall/danger_precision/roc_auc/confusion_matrix numbers that live in
app/config.py's MODEL_METRICS.

Why this exists: those numbers were originally computed by a one-off script
run outside this repo (loaded the .joblib files directly, scored them against
dataset/ml_training_dataset_v2.csv, and the results were hand-pasted into
app/config.py). There was nothing in the repo that would let anyone reproduce
or refresh them later -- e.g. after the next fine-tuning pass -- without
redoing that by hand again. This script is that missing piece.

Usage:
    python -m app.ml.evaluate              # print results, don't touch config.py
    python -m app.ml.evaluate --write      # also patch app/config.py's MODEL_METRICS

Either way, it writes app/ml/eval_metadata.json (timestamp + dataset snapshot
info) so GET /api/models/evaluation-info can report "when were these numbers
last computed" without re-running anything.

Methodology (fixed -- do not change the split without checking with the
project owner first, since every recorded number is based on it):
    train: event_time <= 2024-12-31
    val:   2025-01-01 to 2025-12-31
    test:  event_time >= 2026-01-01   <- only this split is scored

river_level_score: ~25% of test rows (all data_source == "observed_hourly")
have a NaN value here. Filled with 0.3 before scoring -- matches
app/ml/features.py's compute_river_level_score() production fallback (which
never actually emits NaN at inference time), so this reflects real serving
behavior instead of silently dropping a quarter of the test set.

logreg predict_proba: broken under this repo's pinned/installed scikit-learn
for this particular pickled model (AttributeError: 'LogisticRegression'
object has no attribute 'multi_class' -- a version-skew artifact between
whatever scikit-learn trained/pickled it and whatever's installed here; see
requirements.txt's comment on this same issue). Worked around below with a
manual softmax over decision_function's output, verified numerically
identical to a working predict_proba on a test row. If a future scikit-learn
upgrade fixes this natively, prefer predict_proba directly over the
workaround.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.config import BASE_DIR, MODELS_DIR
from app.ml.model_loader import MODEL_REGISTRY

DATASET_PATH = BASE_DIR / "dataset" / "ml_training_dataset_v2.csv"
EVAL_METADATA_PATH = Path(__file__).resolve().parent / "eval_metadata.json"
CONFIG_PATH = BASE_DIR / "app" / "config.py"

# Fixed display/report order -- matches RISK_CLASS_ORDER in the frontend's
# utils/risk.js. Used for confusion_matrix and all metrics except roc_auc.
CLASSES = ["Low", "Medium", "High", "Danger"]
# sklearn's roc_auc_score(multi_class="ovr") requires `labels` sorted
# alphabetically, or it raises ValueError: Parameter 'labels' must be ordered.
ALPHA_CLASSES = sorted(CLASSES)

FEATURE_COLUMNS = [
    "elevation_mean",
    "slope_mean",
    "distance_to_narayani_m",
    "distance_to_rapti_m",
    "nearest_river_distance",
    "nearest_major_river_encoded",
    "within_2km_narayani",
    "within_2km_rapti",
    "lulc_encoded",
    "historical_flood_evidence",
    "flow_accumulation_score",
    "rainfall_24h_mm",
    "rainfall_score",
    "river_level_score",
]

RIVER_LEVEL_SCORE_FALLBACK = 0.3  # matches app/ml/features.py's production fallback


def _load_test_split() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only the columns needed (event_time, risk_class, features) --
    the full CSV has 72 columns; this keeps memory well under what a 2.6M-row
    load of all of them would need."""
    usecols = ["event_time", "risk_class", "data_source", *FEATURE_COLUMNS]
    df = pd.read_csv(DATASET_PATH, usecols=usecols)

    train = df[df["event_time"] <= "2024-12-31"]
    val = df[(df["event_time"] >= "2025-01-01") & (df["event_time"] <= "2025-12-31")]
    test = df[df["event_time"] >= "2026-01-01"].copy()

    nan_river_level = int(test["river_level_score"].isna().sum())
    test["river_level_score"] = test["river_level_score"].fillna(RIVER_LEVEL_SCORE_FALLBACK)

    split_info = {
        "dataset_path": str(DATASET_PATH.relative_to(BASE_DIR)),
        "dataset_row_count": len(df),
        "train_row_count": len(train),
        "val_row_count": len(val),
        "test_row_count": len(test),
        "test_rows_with_nan_river_level_score": nan_river_level,
        "river_level_score_fill_value": RIVER_LEVEL_SCORE_FALLBACK,
    }
    return test, split_info


def _logreg_predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Workaround for predict_proba() being broken on this pickled LogisticRegression
    under this repo's installed scikit-learn (see module docstring). Manual softmax
    over the same linear decision function predict_proba would use internally."""
    try:
        return model.predict_proba(X)
    except AttributeError:
        z = X @ model.coef_.T + model.intercept_  # == model.decision_function(X)
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)


def _evaluate_one_model(model_id: str, test: pd.DataFrame) -> dict[str, Any]:
    entry = MODEL_REGISTRY[model_id]
    payload = joblib.load(MODELS_DIR / entry["file"])
    model = payload["model"]
    features = list(payload["features"])  # this model's own training-order column list

    X = test[features].to_numpy(dtype=float)
    if payload.get("scaler") is not None:
        X = payload["scaler"].transform(X)

    if model_id == "logreg":
        raw_proba = _logreg_predict_proba(model, X)
    else:
        raw_proba = model.predict_proba(X)

    # This model's own class-index order (may differ from CLASSES/ALPHA_CLASSES).
    if payload.get("label_encoder") is not None:
        model_classes = list(payload["label_encoder"].classes_)
    else:
        model_classes = list(model.classes_)

    y_true = test["risk_class"].to_numpy()
    predicted_idx = raw_proba.argmax(axis=1)
    y_pred = np.array([model_classes[i] for i in predicted_idx])

    # Danger-class gate, matching app/ml/model_loader.py's ModelBundle.predict_batch()
    # exactly: models that carry a danger_threshold (currently only xgboost_tuned) are
    # only classified "Danger" when P(Danger) clears that threshold; otherwise the
    # prediction falls back to the best of the *other* classes, even where plain argmax
    # would have picked Danger at a lower probability (see model_loader.py's module
    # docstring for why a threshold above 0.5 can only remove Danger calls, never add
    # ones argmax wouldn't already make). If evaluate.py didn't mirror this, the recorded
    # MODEL_METRICS for xgboost_tuned would describe plain-argmax behavior the deployed
    # model never actually exhibits.
    danger_threshold = payload.get("danger_threshold")
    if danger_threshold is not None and "Danger" in model_classes:
        danger_col = model_classes.index("Danger")
        other_cols = [i for i in range(len(model_classes)) if i != danger_col]
        best_other_idx = raw_proba[:, other_cols].argmax(axis=1)
        best_other_class = np.array([model_classes[other_cols[i]] for i in best_other_idx])
        gate_passes = raw_proba[:, danger_col] >= danger_threshold
        y_pred = np.where(gate_passes, "Danger", best_other_class)

    # Reorder probability columns into ALPHA_CLASSES order specifically for roc_auc_score.
    class_to_col = {cls: raw_proba[:, i] for i, cls in enumerate(model_classes)}
    proba_alpha_order = np.column_stack([class_to_col[c] for c in ALPHA_CLASSES])

    danger_idx = CLASSES.index("Danger")
    recall_per_class = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)

    return {
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=CLASSES)), 4),
        "danger_recall": round(float(recall_per_class[danger_idx]), 4),
        "danger_precision": round(float(precision_per_class[danger_idx]), 4),
        "roc_auc": round(
            float(
                roc_auc_score(
                    y_true, proba_alpha_order, multi_class="ovr", average="macro", labels=ALPHA_CLASSES
                )
            ),
            4,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }


def _format_metrics_block(metrics: dict[str, dict[str, Any]]) -> str:
    """Render metrics as the exact Python source that goes inside app/config.py's
    MODEL_METRICS = {...} assignment -- used by both --write and the printed preview."""
    lines = ["MODEL_METRICS: dict[str, dict[str, Any]] = {"]
    for model_id, m in metrics.items():
        lines.append(f'    "{model_id}": {{')
        lines.append(f'        "test_accuracy": {m["test_accuracy"]},')
        lines.append(f'        "macro_f1": {m["macro_f1"]},')
        lines.append(f'        "danger_recall": {m["danger_recall"]},')
        lines.append(f'        "danger_precision": {m["danger_precision"]},')
        lines.append(f'        "roc_auc": {m["roc_auc"]},')
        lines.append('        "confusion_matrix": [')
        for row in m["confusion_matrix"]:
            lines.append(f"            {row},")
        lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def _write_config(metrics: dict[str, dict[str, Any]]) -> None:
    """Patch app/config.py in place, replacing everything between the
    BEGIN/END MODEL_METRICS markers with freshly generated code."""
    text = CONFIG_PATH.read_text()
    pattern = re.compile(
        r"# BEGIN MODEL_METRICS.*?# END MODEL_METRICS", re.DOTALL
    )
    if not pattern.search(text):
        raise RuntimeError(
            "Could not find '# BEGIN MODEL_METRICS' / '# END MODEL_METRICS' markers "
            f"in {CONFIG_PATH} -- refusing to guess where to write. Add the markers "
            "around the MODEL_METRICS assignment (see app/config.py) and retry."
        )
    new_block = (
        f"# BEGIN MODEL_METRICS (auto-generated by `python -m app.ml.evaluate --write` "
        f"on {datetime.now(timezone.utc).strftime('%Y-%m-%d')} -- do not hand-edit "
        f"within these markers, re-run the script instead)\n"
        f"{_format_metrics_block(metrics)}\n"
        f"# END MODEL_METRICS"
    )
    CONFIG_PATH.write_text(pattern.sub(new_block, text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="Patch app/config.py's MODEL_METRICS in place"
    )
    args = parser.parse_args()

    start = time.monotonic()
    print(f"Loading {DATASET_PATH.relative_to(BASE_DIR)} ...")
    test, split_info = _load_test_split()
    print(
        f"  {split_info['dataset_row_count']:,} total rows -> "
        f"train {split_info['train_row_count']:,} / val {split_info['val_row_count']:,} / "
        f"test {split_info['test_row_count']:,}"
    )
    print(
        f"  {split_info['test_rows_with_nan_river_level_score']:,} test rows had NaN "
        f"river_level_score, filled with {RIVER_LEVEL_SCORE_FALLBACK}"
    )

    metrics: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_REGISTRY:
        print(f"Scoring {model_id} ...")
        metrics[model_id] = _evaluate_one_model(model_id, test)

    elapsed = time.monotonic() - start
    print(f"\nDone in {elapsed:.1f}s.\n")
    print(_format_metrics_block(metrics))

    eval_metadata = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "models_evaluated": list(MODEL_REGISTRY),
        **split_info,
    }
    EVAL_METADATA_PATH.write_text(json.dumps(eval_metadata, indent=2) + "\n")
    print(f"\nWrote {EVAL_METADATA_PATH.relative_to(BASE_DIR)}")

    if args.write:
        _write_config(metrics)
        print(f"Patched {CONFIG_PATH.relative_to(BASE_DIR)}'s MODEL_METRICS.")
    else:
        print("\n(Ran without --write -- app/config.py was NOT modified. "
              "Re-run with --write to apply these numbers, or hand-copy the block above.)")


if __name__ == "__main__":
    main()
