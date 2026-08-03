import pandas as pd
df = pd.read_csv("npl_glide_events.csv")  # replace with your actual GLIDE CSV filename
for y in [2000, 2008, 2010]:
    print(f"--- Year {y} ---")
    print(df[df["year"] == y][["glidenumber", "event_name", "location", "month", "day", "comments"]].to_string())
    print()