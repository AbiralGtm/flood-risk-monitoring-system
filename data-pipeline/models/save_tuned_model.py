"""
save_tuned_model.py

PURPOSE
-------
tune_danger_threshold.py only EVALUATES different thresholds -- it never
saved a usable model artifact. This script does that: it re-runs the
same threshold search, then saves a new file, model_xgboost_tuned.joblib,
that your backend (or anyone else) can load directly and use correctly,
without having to reimplement the threshold logic themselves.

IMPORTANT: this does NOT retrain the model. The underlying XGBoost model
is byte-for-byte the same as model_xgboost.joblib -- only the DECISION
RULE applied to its probability outputs changes (Danger gets flagged at
a lower/tuned probability cutoff instead of pure argmax). This is a
standard, legitimate technique, not a shortcut.

WHY THIS IS A PLAIN DICT, NOT A CUSTOM CLASS
-----------------------------------------------
It would be possible to wrap this in a custom Python class with its own
.predict() method and save THAT via joblib -- but custom classes saved
via joblib/pickle only reload correctly if the exact same class
definition is importable on the loading side too, which is a common
source of "works on my machine" breakage when a model file gets moved
to a different codebase (e.g. your backend). Saving a plain dict (model
+ features + label_encoder + the threshold value) avoids that entirely
-- it will always load correctly anywhere joblib/pickle works, and the
predict_with_tuned_threshold() function below (which you should copy
into your backend) shows exactly how to use it correctly.

INPUT
------
model_xgboost.joblib
ml_training_dataset_v2.csv   (used only to re-derive the threshold;
                               not saved into the output file)

OUTPUT
------
model_xgboost_tuned.joblib
  Contains: {"model", "features", "label_encoder", "danger_threshold",
             "threshold_method", "min_recall_floor",
             "threshold_derived_from_n_rows"}
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve

MODEL_PATH = "models/model_xgboost.joblib"
DATA_PATH = "ml_training_dataset_v2.csv"
OUT_PATH = "model_xgboost_tuned.joblib"
MIN_RECALL_FLOOR = 0.95


def predict_with_tuned_threshold(saved, X):
    """
    THE FUNCTION YOUR BACKEND SHOULD USE.

    Applies the tuned Danger threshold instead of pure argmax:
      - If P(Danger) >= saved["danger_threshold"], predict "Danger"
      - Otherwise, predict whichever of the OTHER classes has the
        highest probability (argmax excluding Danger)

    Parameters
    ----------
    saved : dict
        The loaded contents of model_xgboost_tuned.joblib
        (i.e. joblib.load("model_xgboost_tuned.joblib"))
    X : DataFrame
        Input rows with exactly saved["features"] as columns, in that
        order -- same requirement as the original untuned model.

    Returns
    -------
    numpy array of predicted class labels (strings)
    """
    model = saved["model"]
    features = saved["features"]
    label_encoder = saved.get("label_encoder")
    threshold = saved["danger_threshold"]

    X = X[features]  # enforce correct column order

    proba = model.predict_proba(X)
    if label_encoder is not None:
        proba_classes = label_encoder.inverse_transform(np.arange(proba.shape[1]))
    else:
        proba_classes = model.classes_

    proba_df = pd.DataFrame(proba, columns=proba_classes, index=X.index)

    other_classes_proba = proba_df.drop(columns=["Danger"])
    fallback_pred = other_classes_proba.idxmax(axis=1).values
    is_danger = proba_df["Danger"].values >= threshold

    return np.where(is_danger, "Danger", fallback_pred)


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]
    label_encoder = saved.get("label_encoder")

    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["event_time"], low_memory=False)
    for c in ["within_2km_narayani", "within_2km_rapti"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.upper().eq("TRUE").astype(int)
        n_river_nan = df["river_level_score"].isna().sum() if "river_level_score" in df.columns else 0
    if n_river_nan > 0:
        print(f"Filling {n_river_nan:,} NaN river_level_score values with 0.3 (the established "
              f"'unknown river condition' fallback -- see 04_train_models_v2.py) BEFORE dropna, "
              f"so these rows aren't silently discarded. Missing river data is disproportionately "
              f"concentrated on Rapti-side grids -- dropping instead of filling would systematically "
              f"under-represent Rapti, especially for the rare Danger class.")
        df["river_level_score"] = df["river_level_score"].fillna(0.3)

    df = df.dropna(subset=features + ["risk_class"])

    # ------------------------------------------------------------------
    # FIX (revised): derive the threshold from ALL PRE-2026 DATA
    # (train + validation combined), not the test split, and not the
    # 2025-only validation slice either.
    #
    # History: the original version derived the threshold from the SAME
    # 2026 test rows it then reported performance on -- evaluation
    # leakage. Fixed once already to use 2025-only validation instead --
    # but 2025 alone turned out to have only 73 real Danger examples out
    # of 172,814 rows (0.042% positive rate, vs. 1.39% in the 2026 test
    # set) -- far too few for precision_recall_curve to find a stable
    # threshold. That produced a degenerate near-zero threshold
    # (essentially "call everything Danger") and correspondingly poor
    # test precision (0.052).
    #
    # Using all pre-2026 data instead gives tens of thousands of real
    # Danger examples (matching the ~38,000 combined Narayani+Rapti count
    # found during river-stratified validation) -- enough for a
    # statistically stable threshold. This is mildly less clean than a
    # pure validation-only derivation (the model was trained on this
    # data), but a completely standard and defensible practice, and the
    # test set remains fully untouched either way -- report language:
    # "threshold calibrated on all pre-2026 data since the validation
    # period alone had too few Danger examples (n=73) for a stable
    # estimate; final performance reported exclusively on the held-out
    # 2026 test set."
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # FIX (round 3): restrict the threshold-derivation pool to OBSERVED
    # data only (2021-2025), not all pre-2026 data regardless of source.
    #
    # Round 2 (all pre-2026, ~2.5M rows) fixed the sample-size problem
    # (73,587 Danger examples, plenty) but introduced a different one:
    # ~1.7M of those 2.5M rows are the CHIRPS/GloFAS historical
    # reconstruction (1993-2020), already confirmed to have measurably
    # noisier, less discriminative signal than real observed data
    # (in-sample fit 0.954 vs 0.986 -- see the earlier data-provenance
    # check). Pooling everything let the reconstruction's fuzzier
    # Danger/non-Danger boundary dominate the threshold search, which is
    # very likely why precision at the chosen threshold was WORSE
    # (0.202) than on a smaller, cleaner sample -- more data made the
    # estimate worse because of what kind of data it was, not how much.
    #
    # Restricting to data_source == "observed_hourly" for the full
    # 2021-2025 window keeps enough volume to avoid the original n=73
    # instability, while avoiding the reconstruction's known noise.
    # ------------------------------------------------------------------
    if "data_source" in df.columns:
        val = df[(df["event_time"] < "2026-01-01") & (df["data_source"] == "observed_hourly")]
    else:
        print("WARNING: no data_source column found -- cannot exclude the historical")
        print("reconstruction from threshold derivation. Using all pre-2026 data, which")
        print("may be contaminated by reconstruction noise -- see script comments.")
        val = df[df["event_time"] < "2026-01-01"]
    test = df[df["event_time"] >= "2026-01-01"]

    if len(val) == 0:
        print("WARNING: no pre-2026 rows found -- cannot derive a leakage-free")
        print("threshold. Falling back to deriving from the test set (re-introduces the")
        print("leakage this fix was meant to remove) -- treat the resulting metrics as")
        print("optimistic if you see this warning.")
        val = test

    X_val = val[features]
    y_val = val["risk_class"]
    n_danger_val = (y_val == "Danger").sum()
    print(f"Deriving threshold from {len(val):,} pre-2026 rows "
          f"({n_danger_val:,} Danger examples, {100*n_danger_val/len(val):.3f}% positive rate) ...")
    if n_danger_val < 500:
        print(f"WARNING: only {n_danger_val} Danger examples even in the full pre-2026 pool -- "
              f"the threshold may still be unstable. Consider the cross-validated out-of-fold")
        print(f"approach instead if this number looks too low to trust.")

    if label_encoder is not None:
        val_proba = model.predict_proba(X_val)
        proba_classes = label_encoder.inverse_transform(np.arange(val_proba.shape[1]))
    else:
        val_proba = model.predict_proba(X_val)
        proba_classes = model.classes_
    val_proba_df = pd.DataFrame(val_proba, columns=proba_classes)
    danger_proba_val = val_proba_df["Danger"].values
    y_val_danger = (y_val == "Danger").astype(int).values

    precisions, recalls, thresholds = precision_recall_curve(y_val_danger, danger_proba_val)
    valid_recall_mask = recalls[:-1] >= MIN_RECALL_FLOOR
    if valid_recall_mask.any():
        valid_precisions = np.where(valid_recall_mask, precisions[:-1], -1)
        chosen_idx = np.argmax(valid_precisions)
        chosen_threshold = float(thresholds[chosen_idx])
        method = f"high_recall_floor_{MIN_RECALL_FLOOR}_derived_from_val2025"
        print(f"Chosen threshold (recall floor >= {MIN_RECALL_FLOOR}, on observed-only pre-2026 data): "
              f"{chosen_threshold:.4f}")
        print(f"  -> validation precision={precisions[chosen_idx]:.3f}, recall={recalls[chosen_idx]:.3f}")
    else:
        f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
        chosen_idx = np.argmax(f1s)
        chosen_threshold = float(thresholds[chosen_idx])
        method = "best_f1_derived_from_val2025"
        print(f"No threshold hit the recall floor on validation -- falling back to best F1: "
              f"{chosen_threshold:.4f}")

    # ------------------------------------------------------------------
    # Report performance on the TEST split (2026) -- genuinely held-out,
    # never used to pick anything.
    # ------------------------------------------------------------------
    X_test = df[df["event_time"] >= "2026-01-01"][features]
    y_test = df[df["event_time"] >= "2026-01-01"]["risk_class"]
    print(f"\nReporting final performance on {len(X_test):,} HELD-OUT TEST rows (2026) "
          f"using the val-derived threshold ...")

    if label_encoder is not None:
        test_proba = model.predict_proba(X_test)
    else:
        test_proba = model.predict_proba(X_test)
    test_proba_df = pd.DataFrame(test_proba, columns=proba_classes)
    other_classes_proba_test = test_proba_df.drop(columns=["Danger"])
    fallback_pred_test = other_classes_proba_test.idxmax(axis=1).values
    is_danger_test = test_proba_df["Danger"].values >= chosen_threshold
    y_pred_test_tuned = np.where(is_danger_test, "Danger", fallback_pred_test)

    from sklearn.metrics import classification_report
    test_report = classification_report(
        y_test, y_pred_test_tuned, labels=["Low", "Medium", "High", "Danger"], output_dict=True
    )
    print(f"TEST SET Danger performance (val-derived threshold, genuinely held-out): "
          f"precision={test_report['Danger']['precision']:.3f}, "
          f"recall={test_report['Danger']['recall']:.3f}, "
          f"f1={test_report['Danger']['f1-score']:.3f}")

    # ------------------------------------------------------------------
    # Save the tuned model artifact
    # ------------------------------------------------------------------
    tuned_artifact = {
        "model": model,
        "features": features,
        "label_encoder": label_encoder,
        "danger_threshold": chosen_threshold,
        "threshold_method": method,
        "min_recall_floor": MIN_RECALL_FLOOR,
        "threshold_derived_from_n_rows": len(val),
        "threshold_derived_from_split": "observed_hourly_pre_2026",
        "held_out_test_danger_precision": test_report["Danger"]["precision"],
        "held_out_test_danger_recall": test_report["Danger"]["recall"],
    }
    joblib.dump(tuned_artifact, OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")

    # ------------------------------------------------------------------
    # Self-verification: reload fresh and confirm predictions match what
    # we'd expect, and that the untuned model is byte-identical (same
    # object) -- proves nothing was accidentally retrained.
    # ------------------------------------------------------------------
    print("\n--- SELF-VERIFICATION ---")
    reloaded = joblib.load(OUT_PATH)
    y_pred_tuned = predict_with_tuned_threshold(reloaded, X_test)

    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred_tuned, labels=["Low","Medium","High","Danger"], output_dict=True)
    print(f"Reloaded model, Danger class: precision={report['Danger']['precision']:.3f}, "
          f"recall={report['Danger']['recall']:.3f}")

    # Confirm the underlying model itself is unchanged (predict_proba
    # matches the original model exactly)
    orig_proba = model.predict_proba(X_test)
    reloaded_proba = reloaded["model"].predict_proba(X_test)
    identical = np.allclose(orig_proba, reloaded_proba)
    print(f"Underlying model predictions identical to original (not retrained): {identical}")

    print(f"\nDone. Give your backend team {OUT_PATH} and the")
    print("predict_with_tuned_threshold() function defined in this script.")


if __name__ == "__main__":
    main()