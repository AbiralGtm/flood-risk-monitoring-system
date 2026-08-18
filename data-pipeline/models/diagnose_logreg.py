"""
diagnose_logreg.py

Checks whether model_logreg.joblib's label encoder is consistent with
the other three models' encoders -- the most likely cause of the
0.015 accuracy anomaly (a mismatched encoder would cause inverse_transform
to return the wrong string label for nearly every prediction).
"""

import joblib
import pandas as pd
import numpy as np

logreg_b = joblib.load("model_logreg.joblib")
rf_b = joblib.load("model_random_forest.joblib")
xgbt_b = joblib.load("model_xgboost_tuned.joblib")

def get_le(bundle):
    return bundle.get("label_encoder") if isinstance(bundle, dict) else None

le_logreg = get_le(logreg_b)
le_rf = get_le(rf_b)
le_xgbt = get_le(xgbt_b)

print("=== Label encoder class order per model ===")
print(f"Logistic Regression: {list(le_logreg.classes_) if le_logreg is not None else 'NO ENCODER SAVED'}")
print(f"Random Forest:       {list(le_rf.classes_) if le_rf is not None else 'NO ENCODER SAVED'}")
print(f"XGBoost (tuned):      {list(le_xgbt.classes_) if le_xgbt is not None else 'NO ENCODER SAVED'}")

if le_logreg is not None and le_xgbt is not None:
    if list(le_logreg.classes_) != list(le_xgbt.classes_):
        print("\n*** MISMATCH CONFIRMED: LogReg's label encoder differs from the others. ***")
        print("*** This is very likely the root cause of the broken accuracy. ***")
    else:
        print("\nEncoders match -- the bug is likely elsewhere. See raw prediction check below.")

# Also directly check what raw predictions look like
print("\n=== Raw prediction check ===")
logreg_model = logreg_b["model"] if isinstance(logreg_b, dict) else logreg_b
logreg_feats = logreg_b.get("features") if isinstance(logreg_b, dict) else None
print(f"LogReg features saved: {logreg_feats}")
print(f"LogReg model classes_ (sklearn's own, from training): {logreg_model.classes_}")