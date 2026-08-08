"""
diagnose_rapti_danger_ceiling.py

PURPOSE
-------
evaluate_by_river.py found ZERO Danger examples for Rapti in the entire
test set -- not just rare, literally zero. This could mean two very
different things:

  1. PHYSICALLY REAL: Rapti genuinely never reached Danger-level severity
     during the test period -- a real fact, not a bug.
  2. STRUCTURALLY BROKEN: the global risk_class thresholds (calibrated on
     a COMBINED Narayani+Rapti distribution) may be set so high that
     Rapti's final_risk_score can NEVER cross into Danger, regardless of
     how severe conditions actually are -- because Rapti's scores are
     systematically depressed by its widespread river_level_score
     fallback (0.3, most of the time, per the earlier confirmed 24-47%
     missing-data finding).

This checks which one it is, across the FULL dataset (not just the test
split), by comparing each river's final_risk_score distribution against
the actual Danger threshold.

INPUT
------
ml_training_dataset_v2.csv
(reads the SAME file evaluate_by_river.py used)

OUTPUT
------
Prints a clear diagnosis with supporting numbers.
"""

import pandas as pd
import numpy as np

DATA_PATH = "ml_training_dataset_v2.csv"


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.dropna(subset=["final_risk_score", "nearest_major_river", "risk_class"])
    print(f"Full dataset: {len(df):,} rows\n")

    # ------------------------------------------------------------------
    # 1. Does Rapti have ANY Danger rows anywhere (train+val+test), or is
    #    it zero everywhere, not just in the 2026 test window?
    # ------------------------------------------------------------------
    print("=== Danger-class representation by river, FULL dataset ===")
    danger_by_river = df[df["risk_class"] == "Danger"]["nearest_major_river"].value_counts()
    print(danger_by_river)
    total_by_river = df["nearest_major_river"].value_counts()
    print(f"\nTotal rows by river: {dict(total_by_river)}")
    for river in total_by_river.index:
        n_danger = danger_by_river.get(river, 0)
        pct = 100 * n_danger / total_by_river[river]
        print(f"  {river}: {n_danger:,} Danger rows out of {total_by_river[river]:,} ({pct:.3f}%)")

    # ------------------------------------------------------------------
    # 2. The actual Danger threshold, and each river's score distribution
    #    relative to it -- this is the key diagnostic.
    # ------------------------------------------------------------------
    # Recover the threshold empirically: the minimum final_risk_score
    # among rows actually labeled Danger (should be consistent across
    # the whole dataset, since it's one global threshold).
    danger_rows = df[df["risk_class"] == "Danger"]
    if len(danger_rows):
        danger_threshold = danger_rows["final_risk_score"].min()
        print(f"\n=== Empirical Danger threshold (min final_risk_score among Danger rows) ===")
        print(f"Threshold: {danger_threshold:.4f}")
    else:
        print("\nNo Danger rows found anywhere in the dataset -- cannot recover the threshold this way.")
        danger_threshold = None

    print("\n=== final_risk_score distribution by river ===")
    for river in total_by_river.index:
        river_scores = df[df["nearest_major_river"] == river]["final_risk_score"]
        print(f"\n{river} (n={len(river_scores):,}):")
        print(river_scores.describe())
        if danger_threshold is not None:
            max_score = river_scores.max()
            pct_above = 100 * (river_scores >= danger_threshold).mean()
            print(f"  Max final_risk_score for {river}: {max_score:.4f}")
            print(f"  % of {river} rows at/above the Danger threshold ({danger_threshold:.4f}): {pct_above:.3f}%")
            if max_score < danger_threshold:
                print(f"  *** {river}'s MAXIMUM score never even reaches the Danger threshold. ***")
                print(f"  *** This river is STRUCTURALLY INCAPABLE of producing a Danger label. ***")

    # ------------------------------------------------------------------
    # 3. Root cause check: how much is river_level_score fallback usage
    #    suppressing each river's dynamic_hazard_score / final_risk_score?
    # ------------------------------------------------------------------
    if "label_source" in df.columns:
        fallback_col_candidates = [c for c in df.columns if "fallback" in c.lower()]
    print("\n=== River-level score characteristics by river (root-cause check) ===")
    for river in total_by_river.index:
        river_df = df[df["nearest_major_river"] == river]
        if "river_level_score" in river_df.columns:
            rls = river_df["river_level_score"]
            pct_at_fallback = 100 * (rls == 0.3).mean()
            print(f"\n{river}:")
            print(f"  river_level_score mean: {rls.mean():.4f}, max: {rls.max():.4f}")
            print(f"  % of rows exactly at the 0.3 fallback value: {pct_at_fallback:.1f}%")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    rapti_danger_count = danger_by_river.get("Rapti", 0) if len(danger_rows) else 0
    if rapti_danger_count == 0:
        rapti_scores = df[df["nearest_major_river"] == "Rapti"]["final_risk_score"]
        if danger_threshold is not None and rapti_scores.max() < danger_threshold:
            print("STRUCTURAL PROBLEM CONFIRMED: Rapti's maximum final_risk_score across the")
            print("ENTIRE dataset never reaches the Danger threshold. This is not a fact about")
            print("2026 conditions -- Rapti grids are mathematically incapable of being labeled")
            print("Danger under the current formula/threshold combination, regardless of how")
            print("severe real Rapti flooding gets. This needs a fix (see recommendations")
            print("below), not just a documented limitation.")
        else:
            print("Rapti CAN reach the Danger threshold in principle (some rows do, just not")
            print("in the 2026 test window specifically) -- the zero-in-test-set finding may")
            print("be closer to physically real, though still worth checking how RARE it is")
            print("even in training data.")
    else:
        print(f"Rapti has {rapti_danger_count} Danger rows somewhere in the full dataset --")
        print("the zero-in-test-set finding is specific to the 2026 window, not a structural")
        print("impossibility. Still worth checking if this count is suspiciously low.")

    print("\nIF STRUCTURAL: the fix is to calibrate risk_class thresholds SEPARATELY per")
    print("river (or per-river-normalize final_risk_score before applying one threshold),")
    print("rather than one global threshold derived from a distribution that Narayani")
    print("dominates. This is a real methodology decision to document either way.")


if __name__ == "__main__":
    main()
