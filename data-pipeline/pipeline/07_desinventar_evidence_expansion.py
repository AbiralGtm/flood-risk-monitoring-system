"""
07_desinventar_evidence_expansion.py

PURPOSE
-------
Expands your flood_evidence_bharatpur.csv (currently 18 records) using
DesInventar Nepal records for Chitwan district, targeting your known gap
years: 1993, 2000, 2008, 2010, 2011 (and any other pre-2021 events
DesInventar has that BIPAD's incident API doesn't, since BIPAD only
starts May 2021).

RUN THIS LOCALLY, NOT IN THE CLAUDE SANDBOX.
desinventar.net is not reachable from this sandbox's network allowlist.

MANUAL STEP REQUIRED (DesInventar has no clean REST API like BIPAD)
----------------------------------------------------------------------
1. Go to https://www.desinventar.net/ and select Nepal.
2. Use the Analysis module to query: hazard type = Flood ("Inundacion" /
   "Flood"), admin area = Chitwan district, date range 1990-2020.
3. Export the result table (DesInventar supports CSV/Excel export from
   its query results page).
4. Save it locally as desinventar_chitwan_flood_raw.csv

WHY THIS CAN'T BE FULLY AUTOMATED
------------------------------------
DesInventar's web interface is query-driven, not a stable REST endpoint
like bipadportal.gov.np. This script picks up AFTER you've exported the
raw query result, and does the geocoding/cross-checking/formatting work
that would otherwise be manual.

IMPORTANT LIMITATIONS (per the research report, be honest about these
in your training data and your report)
----------------------------------------------------------------------
- DesInventar gives district/VDC-level location, NOT coordinates. This
  script assigns a district/VDC centroid as a rough proxy coordinate --
  NOT a precise grid-level location. Flag these evidence rows as
  "weak_location" so your history_score spatial join treats them with
  appropriate uncertainty (e.g. spread influence across a wider radius
  than your precise BIPAD-sourced records).
- Pre-2017 VDC names need crosswalking to current Bharatpur wards (the
  2017 federal restructuring merged ~4,000 VDCs into ~6,743 wards
  nationally). This script includes a small manual crosswalk table for
  the VDCs that became Bharatpur Metropolitan City -- VERIFY this table
  against the official Government of Nepal restructuring gazette, since
  an incomplete/wrong crosswalk silently mislabels historical evidence.
- DesInventar records are known to disagree with EM-DAT and MoHA on
  event counts/impacts. Cross-check every candidate event against news
  archives or the MoHA/DRR Portal before promoting it to your training
  evidence file -- do not add DesInventar rows automatically without
  a human sanity check on each one.
"""

import sys
import pandas as pd


# VDCs that were merged into Bharatpur Metropolitan City in the 2017
# restructuring. VERIFY against the official gazette before trusting this --
# this is a starting point, not a certified crosswalk.
VDC_TO_BHARATPUR_WARD_CROSSWALK = {
    "Bharatpur": "multiple wards (former municipality core)",
    "Ratnanagar": "not merged -- separate municipality, do not include",
    "Fulbari": "ward 22-23 area (approximate)",
    "Jutpani": "ward 24-25 area (approximate)",
    "Gunjanagar": "ward 20-21 area (approximate)",
    "Mangalpur": "ward 26 area (approximate) -- matches your EV025 ward 26 record",
    "Meghauli": "ward 28 area (approximate) -- matches known Sept 2020 flood location",
    # Add more as you find them in the DesInventar export. Do not guess --
    # leave unmatched VDCs flagged as "NEEDS_MANUAL_CROSSWALK" (see below).
}

# Rough centroid coordinates (EPSG:4326, WGS84) for the VDCs above, ONLY
# for use as a fallback "weak_location" proxy -- replace with better
# estimates from a Nepal administrative boundary shapefile if you have one.
VDC_ROUGH_CENTROIDS = {
    "Bharatpur": (27.6767, 84.4322),
    "Fulbari": (27.6550, 84.4100),
    "Jutpani": (27.6300, 84.4600),
    "Gunjanagar": (27.6900, 84.3900),
    "Mangalpur": (27.6200, 84.5200),
    "Meghauli": (27.5300, 84.2200),
}

TARGET_YEARS = [1993, 2000, 2008, 2010, 2011]


def crosswalk_vdc(vdc_name):
    vdc_name = str(vdc_name).strip()
    if vdc_name in VDC_TO_BHARATPUR_WARD_CROSSWALK:
        return VDC_TO_BHARATPUR_WARD_CROSSWALK[vdc_name]
    return "NEEDS_MANUAL_CROSSWALK"


def get_rough_centroid(vdc_name):
    return VDC_ROUGH_CENTROIDS.get(str(vdc_name).strip(), (None, None))


def main():
    RAW_PATH = "desinventar_chitwan_flood_raw.csv"
    EXISTING_EVIDENCE_PATH = "flood_evidence_bharatpur.csv"
    OUT_PATH = "desinventar_candidate_evidence.csv"

    try:
        raw = pd.read_csv(RAW_PATH)
    except FileNotFoundError:
        print(f"MISSING INPUT: {RAW_PATH}")
        print("You need to manually export this from desinventar.net first --")
        print("see the docstring at the top of this script for the exact steps.")
        sys.exit(1)

    print(f"Loaded {len(raw)} raw DesInventar records.")

    # DesInventar export column names vary by query -- adjust these to match
    # your actual export. Common DesInventar fields: "Fechayear", "Fechamonth",
    # "Fechaday", "Nombredivi" (admin unit name), "Muertos" (deaths), etc.
    date_cols_present = [c for c in raw.columns if "fecha" in c.lower() or "date" in c.lower()]
    admin_cols_present = [c for c in raw.columns if "divi" in c.lower() or "admin" in c.lower()
                           or "vdc" in c.lower() or "muni" in c.lower()]

    if not date_cols_present or not admin_cols_present:
        print("WARNING: could not auto-detect date/admin columns in your export.")
        print(f"  Columns found: {list(raw.columns)}")
        print("  Edit this script's column mapping manually before proceeding.")
        sys.exit(1)

    print(f"  Detected date column(s): {date_cols_present}")
    print(f"  Detected admin column(s): {admin_cols_present}")

    admin_col = admin_cols_present[0]

    # Try to build a year column from whatever date fields exist
    if "Fechayear" in raw.columns:
        raw["year"] = raw["Fechayear"]
    elif len(date_cols_present) == 1:
        raw["year"] = pd.to_datetime(raw[date_cols_present[0]], errors="coerce").dt.year
    else:
        print("Could not determine event year automatically -- inspect the raw export manually.")
        sys.exit(1)

    candidates = raw.copy()
    candidates["vdc_crosswalk"] = candidates[admin_col].apply(crosswalk_vdc)
    centroids = candidates[admin_col].apply(get_rough_centroid)
    candidates["rough_lat"] = centroids.apply(lambda t: t[0])
    candidates["rough_lon"] = centroids.apply(lambda t: t[1])
    candidates["location_confidence"] = "weak_location"
    candidates["is_target_year"] = candidates["year"].isin(TARGET_YEARS)

    candidates.to_csv(OUT_PATH, index=False)

    # --- Cross-check against existing evidence to avoid duplicates ---
    try:
        existing = pd.read_csv(EXISTING_EVIDENCE_PATH)
        print(f"\nLoaded {len(existing)} existing evidence records for duplicate checking.")
        if "year" in existing.columns:
            existing_years = set(existing["year"].dropna().astype(int))
            overlap_years = existing_years.intersection(set(candidates["year"].dropna().astype(int)))
            if overlap_years:
                print(f"  NOTE: years {sorted(overlap_years)} already have some evidence -- "
                      f"manually check for duplicate events before merging.")
    except FileNotFoundError:
        print(f"\n(Could not load {EXISTING_EVIDENCE_PATH} for duplicate check -- skipping.)")

    # --- Self-verification / summary ---
    print("\n--- SUMMARY ---")
    n_needs_crosswalk = (candidates["vdc_crosswalk"] == "NEEDS_MANUAL_CROSSWALK").sum()
    n_target_year = candidates["is_target_year"].sum()
    print(f"Total candidate records: {len(candidates)}")
    print(f"Records in target gap years {TARGET_YEARS}: {n_target_year}")
    print(f"Records needing manual VDC crosswalk: {n_needs_crosswalk} "
          f"(these have no rough coordinate assigned yet)")

    print(f"\nSaved: {OUT_PATH}")
    print("\nNEXT STEPS (manual, do not skip):")
    print("  1. Open the output CSV and manually verify each is_target_year=True")
    print("     record against news archives or the MoHA/DRR Portal.")
    print("  2. Fill in NEEDS_MANUAL_CROSSWALK rows with proper ward assignments")
    print("     using the official 2017 restructuring gazette.")
    print("  3. Only THEN append verified rows to flood_evidence_bharatpur.csv,")
    print("     following your existing EV0xx numbering, with location_confidence")
    print("     carried through so history_score spatial join can weight them")
    print("     appropriately (wider radius / lower weight than BIPAD-precise rows).")
    print("  4. If you can pair a target-year event with a CHIRPS/GloFAS")
    print("     reconstructed date from script 06, promote it from history_score-only")
    print("     to a full training row -- otherwise it stays as spatial evidence only,")
    print("     per your own documented rule for pre-BIPAD evidence.")


if __name__ == "__main__":
    main()
