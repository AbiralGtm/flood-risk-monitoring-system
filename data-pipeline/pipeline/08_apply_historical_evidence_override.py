"""
08_apply_historical_evidence_override.py

PURPOSE
-------
Your own empirical testing (yearly-max dynamic_hazard_score ranking across
the 1993-2020 GloFAS/CHIRPS reconstruction) confirmed what your research
report already predicted: GloFAS discharge reconstruction at Devghat
under-discriminates peak severity. Known catastrophic flood years ranked:
  1993: 23/28,  2000: 16/28,  2008: 13/28,  2010: 1/28,  2011: 21/28
Only 2010 stood out. The other four known flood years scored
indistinguishably from ordinary years (full 28-year spread was only
0.032 wide, 0.7701-0.8023) -- this is a genuine reconstruction limitation,
not something further formula-tweaking should try to paper over.

This script does NOT try to make the reconstruction magically discriminate
better. Instead, it applies the same principle your own pipeline already
uses in 02_rainfall_event_selection.py (force-including flood evidence
dates regardless of what the automated severity search finds) -- for
dates you have independent historical confirmation of severe flooding,
the LABEL comes from that historical evidence, not from a reconstruction
already shown to under-detect exactly these events.

WHAT THIS DOES
---------------
For each flood evidence record whose date falls in 1993-2020 (pre-BIPAD,
your documented "history_score-only" era), searches historical_reconstructed
rows within a +/-2 day window (same window logic as 02's evidence-date
forcing), and overrides dynamic_hazard_score for ALL matched rows
(across all rainfall stations for that date) to a floor value guaranteed
to rank above every value the reconstruction itself produced across the
full 28-year record (so these dates classify as High/Danger once combined
with grid-level static scores in the eventual training set, instead of
landing wherever the reconstruction's under-discrimination happens to
place them).

Original computed values are preserved in a separate column
(dynamic_hazard_score_reconstructed) for full transparency -- nothing is
silently overwritten, and you can report exactly which rows were
overridden and why.

INPUT
------
historical_reconstructed_1993_2020.csv  (output of 06)
flood_evidence_bharatpur.csv            (your evidence file)

OUTPUT
------
historical_reconstructed_1993_2020_overridden.csv
  Same rows as input, PLUS:
    dynamic_hazard_score_reconstructed : the ORIGINAL computed value,
                                          preserved for transparency
    label_source : "historical_evidence_override" or "reconstructed"
    override_evidence_id : which evidence record triggered the override
                            (blank if not overridden)
  dynamic_hazard_score itself is updated to the floor value for
  overridden rows, unchanged for all others.
"""

import sys
import numpy as np
import pandas as pd

RECONSTRUCTED_PATH = "historical_reconstructed_1993_2020.csv"
EVIDENCE_PATH = "flood_evidence_bharatpur.csv"
OUT_PATH = "historical_reconstructed_1993_2020_overridden.csv"

SEARCH_WINDOW_DAYS = 2   # matches the +/-2 day window 02_rainfall_event_selection.py
                          # already uses when force-including evidence dates

# Confidence-to-floor mapping. The floor must exceed the maximum value the
# reconstruction itself produced across the full 1993-2020 record (0.8023,
# empirically confirmed) so overridden rows reliably rank as the most
# severe conditions once combined with grid-level static scores -- these
# are, after all, dates independently confirmed as major floods.
CONFIDENCE_FLOOR = {
    "High": 0.95,
    "Medium": 0.90,
    "Low": 0.85,
}
DEFAULT_FLOOR = 0.85  # for evidence rows with no/unrecognized confidence_level


def main():
    try:
        recon = pd.read_csv(RECONSTRUCTED_PATH, parse_dates=["event_time"])
    except FileNotFoundError:
        print(f"MISSING INPUT: {RECONSTRUCTED_PATH}")
        sys.exit(1)

    try:
        evidence = pd.read_csv(EVIDENCE_PATH, parse_dates=["event_date"])
    except FileNotFoundError:
        print(f"MISSING INPUT: {EVIDENCE_PATH}")
        sys.exit(1)

    # Only pre-2021 evidence needs this override -- your 2021+ rows already
    # come from real BIPAD hourly data with genuine river-level readings,
    # not a reconstruction, so they don't have this limitation.
    evidence = evidence.copy()
    evidence["event_date"] = pd.to_datetime(evidence["event_date"]).dt.tz_localize(
        "UTC", nonexistent="NaT", ambiguous="NaT"
    ) if evidence["event_date"].dt.tz is None else evidence["event_date"]
    pre_2021 = evidence[evidence["event_date"] < "2021-01-01"].copy()
    print(f"Loaded {len(evidence)} total evidence records, "
          f"{len(pre_2021)} are pre-2021 (candidates for override).")

    if len(pre_2021) == 0:
        print("No pre-2021 evidence records found -- nothing to override. "
              "Check EVIDENCE_PATH and your evidence file's event_date column.")
        sys.exit(0)

    recon["dynamic_hazard_score_reconstructed"] = recon["dynamic_hazard_score"]
    recon["label_source"] = "reconstructed"
    recon["override_evidence_id"] = ""

    original_max_by_year = recon.groupby(recon["event_time"].dt.year)["dynamic_hazard_score"].max()
    print(f"\nBEFORE override -- max dynamic_hazard_score across all 28 years: "
          f"{original_max_by_year.max():.4f}")

    total_rows_overridden = 0
    dates_overridden = []

    for _, ev in pre_2021.iterrows():
        ev_date = ev["event_date"]
        confidence = str(ev.get("confidence_level", "")).strip()
        floor = CONFIDENCE_FLOOR.get(confidence, DEFAULT_FLOOR)
        evidence_id = ev.get("evidence_id", ev_date.date())
        date_precision = str(ev.get("date_precision", "exact")).strip().lower()

        if date_precision == "year_only":
            # Don't trust the placeholder date at all -- search the WHOLE
            # year for whichever day the reconstruction itself already
            # scores highest, and override that real day. This uses
            # actual data to find the date instead of guessing one.
            year = int(ev.get("event_year", ev_date.year))
            year_mask = recon["event_time"].dt.year == year
            year_rows = recon.loc[year_mask]

            if len(year_rows) == 0:
                print(f"  WARNING: evidence {evidence_id} (year_only, {year}) has NO "
                      f"reconstructed data for that year at all -- check your CHIRPS/"
                      f"GloFAS pull covers this year.")
                continue

            peak_time = year_rows.loc[
                year_rows["dynamic_hazard_score_reconstructed"].idxmax(), "event_time"
            ]
            window_start = peak_time - pd.Timedelta(days=SEARCH_WINDOW_DAYS)
            window_end = peak_time + pd.Timedelta(days=SEARCH_WINDOW_DAYS)
            mask = (recon["event_time"] >= window_start) & (recon["event_time"] <= window_end)
            print(f"  {evidence_id} (year_only, {year}): identified {peak_time.date()} as the "
                  f"reconstruction's own highest-scoring day that year -- overriding a "
                  f"+/-{SEARCH_WINDOW_DAYS} day window around it.")
        else:
            window_start = ev_date - pd.Timedelta(days=SEARCH_WINDOW_DAYS)
            window_end = ev_date + pd.Timedelta(days=SEARCH_WINDOW_DAYS)
            mask = (recon["event_time"] >= window_start) & (recon["event_time"] <= window_end)

        n_matched = mask.sum()

        if n_matched == 0:
            print(f"  WARNING: evidence {evidence_id} ({ev_date.date()}) has NO matching "
                  f"rows in the reconstructed data within +/-{SEARCH_WINDOW_DAYS} days -- "
                  f"check that your CHIRPS/GloFAS pull actually covers this date.")
            continue

        # Only raise the score, never lower it -- if the reconstruction
        # already scored something higher than the floor for this window
        # (unlikely given what we found, but possible), respect that.
        before_vals = recon.loc[mask, "dynamic_hazard_score"].copy()
        recon.loc[mask, "dynamic_hazard_score"] = np.maximum(before_vals, floor)
        recon.loc[mask, "label_source"] = "historical_evidence_override"
        recon.loc[mask, "override_evidence_id"] = str(evidence_id)

        total_rows_overridden += n_matched
        dates_overridden.append((evidence_id, ev_date.date(), confidence, floor, n_matched))

    print(f"\nOverrode {total_rows_overridden} rows across {len(dates_overridden)} evidence dates:")
    for evidence_id, date, confidence, floor, n in dates_overridden:
        print(f"  {evidence_id} ({date}, confidence={confidence or 'unknown'}) "
              f"-> floor {floor}, {n} rows affected")

    new_max_by_year = recon.groupby(recon["event_time"].dt.year)["dynamic_hazard_score"].max()
    print(f"\nAFTER override -- yearly max dynamic_hazard_score, top 15:")
    print(new_max_by_year.sort_values(ascending=False).head(15))

    print(f"\nRank check for known flood years after override:")
    ranked = new_max_by_year.sort_values(ascending=False)
    for y in [1993, 2000, 2008, 2010, 2011]:
        if y in ranked.index:
            rank = list(ranked.index).index(y) + 1
            print(f"  {y}: rank {rank}/28, score {ranked[y]:.4f}")
        else:
            print(f"  {y}: no data in reconstructed file")

    recon.to_csv(OUT_PATH, index=False)

    print("\n--- SELF-VERIFICATION ---")
    checks_passed = True
    if recon["dynamic_hazard_score"].between(0, 1).all():
        print("PASS: dynamic_hazard_score still within [0,1] after override")
    else:
        print("FAIL: dynamic_hazard_score out of range after override")
        checks_passed = False

    n_overridden_rows = (recon["label_source"] == "historical_evidence_override").sum()
    if n_overridden_rows == total_rows_overridden:
        print(f"PASS: {n_overridden_rows} rows correctly tagged as overridden")
    else:
        print(f"FAIL: tagged count ({n_overridden_rows}) doesn't match expected ({total_rows_overridden})")
        checks_passed = False

    print(f"\nOverall: {'PASS' if checks_passed else 'FAIL -- review warnings above'}")
    print(f"Saved: {OUT_PATH}")
    print("\nMETHODOLOGY NOTE for your report:")
    print("Dynamic hazard scores for grid-cell-independent, gauge-level GloFAS/CHIRPS")
    print("reconstruction were found to under-discriminate known major flood years")
    print("(1993, 2000, 2008, 2011 ranked 23rd, 16th, 13th, and 21st respectively out")
    print("of 28 reconstructed years, with only 2010 standing out) -- consistent with")
    print("documented GloFAS peak-discharge underestimation at this gauge. For dates")
    print("with independent historical confirmation of major flooding, the training")
    print("label was set from that historical evidence directly (floor value by")
    print("evidence confidence level) rather than from the reconstruction alone,")
    print("following the same evidence-forcing principle already used for BIPAD-era")
    print("event selection. The original reconstructed values are preserved")
    print("(dynamic_hazard_score_reconstructed) for transparency and comparison.")


if __name__ == "__main__":
    main()
