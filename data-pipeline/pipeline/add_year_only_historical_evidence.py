"""
add_year_only_historical_evidence.py

PURPOSE
-------
Adds the 5 documented major Narayani flood years (1993, 2000, 2008, 2010,
2011 -- per Kafle 2020, cited across multiple papers in your literature
review) to flood_evidence_bharatpur.csv, WITHOUT fabricating day-level
dates that a search couldn't find. These are added as "year_only"
precision entries -- the flood's YEAR is well-documented, the exact DAY
is not (and inventing one would be dishonest data).

WHY NOT JUST PICK A PLACEHOLDER DATE
--------------------------------------
Picking e.g. "August 15" for each year and searching +/-2 days around it
(like your other evidence) would silently treat a guess as a fact. This
script instead adds them as year_only entries, and the companion updated
08_apply_historical_evidence_override.py searches the ENTIRE year for
whichever day the reconstruction itself scores highest, and overrides
THAT actual day -- using real reconstructed data to find the date,
rather than inventing one.

LOCATION CAVEAT
-----------------
Coordinates below are set to Mangalpur (ward 26 area), which recurs
repeatedly across multiple documented Narayani flood events (including
your existing EV025 Sept 2020 record) as a commonly-affected settlement.
This is a reasonable representative point for "Narayani flooded
Bharatpur/Chitwan," NOT a verified location for these specific 5
historical events. Flagged confidence_level=Medium reflects: high
confidence the FLOOD OCCURRED (multiple independent literature sources
agree), lower confidence in the exact location/date (hence "settlement"
not "exact" spatial_accuracy, and date_precision=year_only).

RECOMMENDATION
---------------
Treat these as a documented placeholder, not a finished data source.
Prioritize the DesInventar Nepal data pull (07_desinventar_evidence_expansion.py)
to potentially replace these with better-sourced day-level dates and
locations if DesInventar's records for these years turn out to be more
specific.

INPUT
------
flood_evidence_bharatpur.csv (your existing evidence file)

OUTPUT
------
flood_evidence_bharatpur.csv (same file, 5 new rows appended)
"""

import sys
import pandas as pd

EVIDENCE_PATH = "flood_evidence_bharatpur.csv"

# Representative coordinate: Mangalpur area, Bharatpur (recurs across
# multiple documented Narayani flood events at similar wards/settlements)
REPRESENTATIVE_LAT = 27.6200
REPRESENTATIVE_LON = 84.5200

NEW_EVIDENCE = [
    {
        "event_year": 1993,
        "notes": "One of 5 major Narayani floods in the last 3 decades per Kafle (2020); "
                 "badly affected both Chitwan and Nawalparasi districts. Day-level date "
                 "not found via available search -- year confirmed via literature only.",
    },
    {
        "event_year": 2000,
        "notes": "One of 5 major Narayani floods in the last 3 decades per Kafle (2020). "
                 "Day-level date not found via available search -- year confirmed via "
                 "literature only.",
    },
    {
        "event_year": 2008,
        "notes": "One of 5 major Narayani floods in the last 3 decades per Kafle (2020). "
                 "Day-level date not found via available search -- year confirmed via "
                 "literature only. NOTE: do not confuse with the well-documented Aug 18, "
                 "2008 Bihar/Koshi embankment breach -- that is a different river "
                 "(Koshi, not Narayani) and a different event.",
    },
    {
        "event_year": 2010,
        "notes": "One of 5 major Narayani floods in the last 3 decades per Kafle (2020). "
                 "Day-level date not found via available search -- year confirmed via "
                 "literature only.",
    },
    {
        "event_year": 2011,
        "notes": "One of 5 major Narayani floods in the last 3 decades per Kafle (2020); "
                 "badly affected both Chitwan and Nawalparasi districts. Day-level date "
                 "not found via available search -- year confirmed via literature only.",
    },
]


def main():
    try:
        existing = pd.read_csv(EVIDENCE_PATH)
    except FileNotFoundError:
        print(f"MISSING INPUT: {EVIDENCE_PATH}")
        sys.exit(1)

    print(f"Loaded {len(existing)} existing evidence records.")

    # Continue your existing EV0xx numbering
    existing_ids = existing["evidence_id"].astype(str) if "evidence_id" in existing.columns else pd.Series([])
    existing_nums = [int(''.join(filter(str.isdigit, str(x)))) for x in existing_ids if any(c.isdigit() for c in str(x))]
    next_num = max(existing_nums, default=0) + 1

    # Warn if any of these years already have SOME evidence (avoid true duplicates)
    if "event_date" in existing.columns:
        existing_years = pd.to_datetime(existing["event_date"], errors="coerce").dt.year.dropna().astype(int)
        for row in NEW_EVIDENCE:
            if row["event_year"] in existing_years.values:
                print(f"  NOTE: year {row['event_year']} already has some evidence in the file -- "
                      f"check for duplication before proceeding.")

    new_rows = []
    for row in NEW_EVIDENCE:
        evidence_id = f"EV{next_num:03d}"
        next_num += 1
        new_rows.append({
            "evidence_id": evidence_id,
            "event_date": f"{row['event_year']}-01-01",  # placeholder for year extraction ONLY --
                                                            # NOT a claim about the actual flood date
            "event_year": row["event_year"],
            "date_precision": "year_only",
            "latitude": REPRESENTATIVE_LAT,
            "longitude": REPRESENTATIVE_LON,
            "spatial_accuracy": "settlement",
            "confidence_level": "Medium",
            "source": "Kafle (2020), cited across multiple secondary literature sources",
            "notes": row["notes"],
        })

    new_df = pd.DataFrame(new_rows)

    # Add any missing columns to the existing dataframe so concat doesn't
    # silently drop data or misalign columns
    for col in new_df.columns:
        if col not in existing.columns:
            existing[col] = pd.NA
            print(f"  Added new column '{col}' to existing evidence schema (was not present before).")

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(EVIDENCE_PATH, index=False)

    print(f"\nAdded {len(new_rows)} year_only evidence records:")
    for r in new_rows:
        print(f"  {r['evidence_id']}: year {r['event_year']}, confidence={r['confidence_level']}, "
              f"date_precision={r['date_precision']}")

    print(f"\nTotal evidence records now: {len(combined)}")
    print(f"\nSaved: {EVIDENCE_PATH}")
    print("\nNEXT STEP: run the updated 08_apply_historical_evidence_override.py, which now")
    print("handles date_precision='year_only' rows by searching the FULL YEAR for the")
    print("actual highest-scoring reconstructed day, rather than guessing a date.")


if __name__ == "__main__":
    main()
