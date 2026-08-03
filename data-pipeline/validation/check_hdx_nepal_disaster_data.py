"""
check_hdx_nepal_disaster_data.py

PURPOSE
-------
Inspects whatever CSV you downloaded from HDX ("Nepal - Disaster data
from 1991-2010") to answer the real question: does this file actually
have day-level dates and locations for Chitwan/Narayani floods, or just
yearly counts? Run this BEFORE assuming the file solves your date-
precision problem for 1993/2000/2008/2010.

USAGE
------
1. Put the downloaded CSV in this same folder.
2. Edit FILE_PATH below to match its actual filename (HDX exports often
   have long auto-generated names -- check what you actually downloaded).
3. Run: python check_hdx_nepal_disaster_data.py
"""

import sys
import pandas as pd

FILE_PATH = "npl_glide_events.csv"  # EDIT to match your actual downloaded filename

TARGET_YEARS = [1993, 2000, 2008, 2010]
CHITWAN_KEYWORDS = ["chitwan", "bharatpur", "narayani", "narayangadh", "narayanghat"]


def main():
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"MISSING: {FILE_PATH}")
        print("Edit FILE_PATH at the top of this script to match whatever filename")
        print("you actually downloaded from HDX, and make sure it's in this same folder.")
        sys.exit(1)
    except Exception as e:
        print(f"Could not read {FILE_PATH} as CSV: {e}")
        print("If it downloaded as .xlsx instead, use: pd.read_excel(FILE_PATH)")
        sys.exit(1)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.\n")
    print("=== ALL COLUMN NAMES (this tells you what precision is available) ===")
    for col in df.columns:
        print(f"  {col}")
    print()

    # Look for date-related columns
    date_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["date", "year", "month", "day", "fecha"]
    )]
    print(f"=== DATE-RELATED COLUMNS FOUND: {date_cols} ===\n")

    # Look for location-related columns
    loc_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["district", "location", "place", "divi", "admin", "muni", "vdc"]
    )]
    print(f"=== LOCATION-RELATED COLUMNS FOUND: {loc_cols} ===\n")

    # Look for hazard-type columns
    hazard_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["hazard", "event", "disaster", "type"]
    )]
    print(f"=== HAZARD-TYPE COLUMNS FOUND: {hazard_cols} ===\n")

    # Try to find Chitwan/Narayani-related rows regardless of exact column name,
    # by searching all text columns for the keywords
    text_cols = df.select_dtypes(include=["object", "str"]).columns
    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        mask |= df[col].astype(str).str.lower().str.contains(
            "|".join(CHITWAN_KEYWORDS), na=False
        )

    chitwan_rows = df[mask]
    print(f"=== ROWS MENTIONING CHITWAN/BHARATPUR/NARAYANI (any column): {len(chitwan_rows)} ===")
    if len(chitwan_rows) > 0:
        print(chitwan_rows.head(20).to_string())
    else:
        print("None found -- this file may not have district/location text at all, "
              "or uses a different naming convention (check LOCATION-RELATED COLUMNS above).")
    print()

    # If we found a year column, check specifically for the 4 target years
    year_col_candidates = [c for c in df.columns if "year" in c.lower()]
    if year_col_candidates:
        year_col = year_col_candidates[0]
        print(f"=== USING '{year_col}' AS YEAR COLUMN ===")
        for y in TARGET_YEARS:
            matches = df[df[year_col].astype(str).str.contains(str(y), na=False)]
            print(f"  {y}: {len(matches)} total rows in the file for this year "
                  f"(across ALL of Nepal, not just Chitwan)")
    else:
        print("No obvious 'year' column found -- check the DATE-RELATED COLUMNS list "
              "above and inspect manually; the year might be embedded in a full date "
              "string instead of its own column.")

    print("\n=== WHAT TO LOOK FOR ===")
    print("GOOD (usable for day-precise evidence): a date column with actual")
    print("  day-month-year values, AND rows above show Chitwan/Narayani-specific")
    print("  flood entries for 1993/2000/2008/2010.")
    print("LIMITED (same as before): only year-level data, or no Chitwan-specific")
    print("  rows at all -- in that case this file doesn't improve on what you")
    print("  already have, and the year_only override approach already built")
    print("  remains your best option until DesInventar's live site is back up.")


if __name__ == "__main__":
    main()
