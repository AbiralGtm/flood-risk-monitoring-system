import pandas as pd

df = pd.read_csv('ml_training_dataset.csv', parse_dates=['event_time'])

print("1. Row count sane?", len(df), "(expect ~2400 grids x ~330 events = ~790k)")
print("2. No duplicate grid+event rows:", df.duplicated(subset=['grid_id','event_id']).sum(), "(should be 0)")
print("3. Missing values in key columns:")
print(df[['elevation_mean','rainfall_score','static_susceptibility_score','final_risk_score']].isna().sum())
print("4. Score ranges (all should be 0-1):")
for c in ['static_susceptibility_score','dynamic_hazard_score','final_risk_score']:
    print(f"   {c}: min={df[c].min():.3f} max={df[c].max():.3f}")
print("5. Calm-day check (should be ~0%):")
calm = df[(df['rainfall_score']<0.05) & (df['river_level_score'].fillna(0)<0.3)]
print(f"   {calm['flood_label'].mean():.1%} flagged High/Danger on calm days")
print("6. risk_class distribution:")
print(df['risk_class'].value_counts())
