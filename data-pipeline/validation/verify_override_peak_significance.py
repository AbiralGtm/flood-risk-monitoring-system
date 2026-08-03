"""
verify_override_peak_significance.py

PURPOSE
-------
Addresses a real gap in 08_apply_historical_evidence_override.py: that
script picks the single highest-dynamic_hazard_score day in each target
year and treats it as "the flood day" -- but never checks whether that
day is a genuine, meaningful standout, or just an arbitrary winner among
many nearly-identical days (which would make the specific date choice
essentially noise, not signal).

This does NOT require any external data source -- it's a direct
statistical check against your own reconstructed data, and it directly
answers: "how much should we trust this specific date?"

WHAT IT COMPUTES, per target year
------------------------------------
1. The identified peak day's dynamic_hazard_score
2. That year's mean and standard deviation of dynamic_hazard_score
   (excluding the peak day itself)
3. The peak day's z-score (how many standard deviations above the
   year's typical day it sits)
4. How many OTHER days that year come within 5% of the peak value
   (if many days are this close, the "peak" is not a clear standout)
5. The peak day's percentile rank within its own year

INTERPRETATION
---------------
z-score > 2 and few near-ties: the peak is a genuine, statistically
  distinct standout -- reasonable to trust as "probably the real event,"
  though still not externally confirmed.
z-score < 1 or many near-ties: the peak is NOT distinguishable from
  ordinary day-to-day variation -- the specific date should be treated
  as essentially unknown, and the override arguably should NOT pin a
  specific narrow date/window at all. Consider widening the override
  window to the full monsoon season (June-September) rather than a
  narrow +/-2 days around one arbitrarily-selected day, OR keeping the
  override at the YEAR level only (e.g. via a year-level feature/flag)
  rather than claiming day-level precision you don't actually have.

INPUT
------
historical_reconstructed_1993_2020.csv (the ORIGINAL, pre-override file
  -- uses dynamic_hazard_score_reconstructed if present, else
  dynamic_hazard_score, so this works whether you run it before or
  after 08_apply_historical_evidence_override.py)
flood_evidence_bharatpur.csv (to identify which years are year_only)
"""

import sys
import numpy as np
import pandas as pd

RECON_PATH = "historical_reconstructed_1993_2020.csv"
EVIDENCE_PATH = "flood_evidence_bharatpur.csv"
NEAR_TIE_THRESHOLD = 0.05  # within 5% of the peak counts as a "near-tie"


def main():
    try:
        recon = pd.read_csv(RECON_PATH, parse_dates=["event_time"])
    except FileNotFoundError:
        print(f"MISSING INPUT: {RECON_PATH}")
        sys.exit(1)

    score_col = "dynamic_hazard_score_reconstructed" if \
        "dynamic_hazard_score_reconstructed" in recon.columns else "dynamic_hazard_score"
    print(f"Using column: {score_col}\n")

    try:
        evidence = pd.read_csv(EVIDENCE_PATH, parse_dates=["event_date"])
        year_only = evidence[evidence.get("date_precision", "") == "year_only"]
        target_years = sorted(year_only["event_year"].dropna().astype(int).unique()) if \
            "event_year" in year_only.columns else []
    except FileNotFoundError:
        target_years = []

    if not target_years:
        print("Could not find year_only evidence records -- checking all 28 years instead.")
        target_years = sorted(recon["event_time"].dt.year.unique())

    print(f"{'Year':<6}{'Peak Date':<14}{'Peak Score':<12}{'Year Mean':<11}"
          f"{'Year Std':<10}{'Z-score':<10}{'Near-ties':<10}{'Percentile'}")
    print("-" * 90)

    results = []
    for year in target_years:
        year_data = recon[recon["event_time"].dt.year == year]
        if len(year_data) == 0:
            print(f"{year:<6}NO DATA FOUND FOR THIS YEAR")
            continue

        daily = year_data.groupby(year_data["event_time"].dt.date)[score_col].max()

        peak_date = daily.idxmax()
        peak_score = daily.max()
        rest = daily.drop(peak_date)
        year_mean = rest.mean()
        year_std = rest.std()
        z_score = (peak_score - year_mean) / year_std if year_std > 0 else float("inf")

        near_tie_threshold_value = peak_score * (1 - NEAR_TIE_THRESHOLD)
        n_near_ties = (rest >= near_tie_threshold_value).sum()

        percentile = (daily < peak_score).mean() * 100

        results.append({
            "year": year, "peak_date": peak_date, "peak_score": peak_score,
            "year_mean": year_mean, "year_std": year_std, "z_score": z_score,
            "n_near_ties": n_near_ties, "percentile": percentile,
        })

        print(f"{year:<6}{str(peak_date):<14}{peak_score:<12.4f}{year_mean:<11.4f}"
              f"{year_std:<10.4f}{z_score:<10.2f}{n_near_ties:<10}{percentile:.1f}%")

    print("\n=== INTERPRETATION ===")
    for r in results:
        year = r["year"]
        if r["z_score"] > 2 and r["n_near_ties"] <= 2:
            verdict = "STANDOUT -- reasonable to trust this specific date"
        elif r["z_score"] > 1 and r["n_near_ties"] <= 5:
            verdict = "MODERATE -- some confidence, but not a sharp standout"
        else:
            verdict = "WEAK -- this date is NOT statistically distinguishable " \
                      "from an ordinary day; treat the day-level pick as unreliable"
        print(f"  {year}: {verdict} (z={r['z_score']:.2f}, {r['n_near_ties']} near-ties)")

    n_weak = sum(1 for r in results if r["z_score"] <= 1 or r["n_near_ties"] > 5)
    print(f"\n{n_weak} of {len(results)} target years have WEAK day-level confidence.")
    if n_weak > 0:
        print("\nRECOMMENDATION for weak years: don't override a narrow +/-2 day window.")
        print("Either (a) widen the override to the full monsoon season for that year")
        print("(June-September) at a LOWER floor value (reflecting genuine uncertainty),")
        print("or (b) keep these as year_only evidence for history_score purposes only,")
        print("without forcing a specific day-level dynamic_hazard_score override at all --")
        print("document this honestly as 'year confirmed, day unknown' in your report.")


if __name__ == "__main__":
    main()
