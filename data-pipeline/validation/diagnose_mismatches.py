import pandas as pd

df = pd.read_csv('ml_training_dataset.csv', parse_dates=['event_time'])
results = pd.read_csv('real_scenario_validation.csv')

# For each row that had valid data, show the ACTUAL rainfall/river conditions
# used at that exact matched grid+event -- this tells us if the model saw
# genuinely calm conditions (data-matching gap) or genuinely severe
# conditions and still said Low (real model problem).
for _, r in results.iterrows():
    if pd.isna(r.get('nearest_grid')):
        continue
    row = df[(df['grid_id'] == r['nearest_grid']) & (df['event_time'] == r['closest_event_used'])]
    if len(row):
        row = row.iloc[0]
        print(f"{r['evidence_id']} ({r['location']}): predicted={r['model_predicted']}")
        print(f"   rainfall_score={row.get('rainfall_score'):.3f}  "
              f"rainfall_24h_mm={row.get('rainfall_24h_mm'):.1f}  "
              f"water_level_m={row.get('water_level_m')}  "
              f"static_susceptibility={row.get('static_susceptibility_score'):.3f}  "
              f"final_risk_score={row.get('final_risk_score'):.3f}")
        print()
