"""
check_evidence_grid_count.py

Directly counts how many grids actually have non-zero historical_flood_evidence
in your real static grid dataset -- run this instead of trusting any estimate.

INPUT: static_grid_dataset_with_stations.csv (or whatever your actual static
       grid file is named -- adjust IN_CSV below if different)
"""

import pandas as pd

IN_CSV = "static_grid_dataset_with_stations.csv"  # adjust if your real filename differs

df = pd.read_csv(IN_CSV)

if "historical_flood_evidence" not in df.columns:
    print(f"Column 'historical_flood_evidence' not found. Available columns:")
    print(list(df.columns))
else:
    total_grids = len(df)
    n_nonzero = (df["historical_flood_evidence"] > 0).sum()
    n_zero = total_grids - n_nonzero

    print(f"Total grids: {total_grids:,}")
    print(f"Grids WITH flood evidence (historical_flood_evidence > 0): {n_nonzero:,} "
          f"({100*n_nonzero/total_grids:.1f}%)")
    print(f"Grids WITHOUT flood evidence: {n_zero:,} ({100*n_zero/total_grids:.1f}%)")

    print(f"\nValue distribution:")
    print(df["historical_flood_evidence"].describe())

    print(f"\nIf historical_flood_evidence is a continuous [0,1] score rather than a")
    print(f"simple count/flag, also check how many grids exceed a few sample cutoffs:")
    for cutoff in [0.1, 0.3, 0.5]:
        n_above = (df["historical_flood_evidence"] > cutoff).sum()
        print(f"  > {cutoff}: {n_above:,} grids")