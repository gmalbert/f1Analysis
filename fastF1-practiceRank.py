import fastf1
import pandas as pd

# Enable cache (optional, but recommended)
fastf1.Cache.enable_cache('f1_cache')

# Choose your session
year = 2024
gp = 'Bahrain'
session_type = 'FP1'

# Load session
session = fastf1.get_session(year, gp, session_type)
session.load()

# Get all laps
laps = session.laps.copy()

# Only keep laps with a valid lap time
laps = laps[laps['LapTime'].notnull()]

# Sort by lap time (ascending) within each lap number
laps['LapPosition'] = laps.groupby('LapNumber')['LapTime'].rank(method='min')

# If you want the overall position for each lap (regardless of lap number), sort by LapTime for the whole session:
laps['OverallPosition'] = laps['LapTime'].rank(method='min')

# Show a few columns
print(laps[['Driver', 'LapNumber', 'LapTime', 'LapPosition', 'OverallPosition']].head(20))

# Save to CSV if you want
laps.to_csv('practice_laps_with_positions.csv', index=False)