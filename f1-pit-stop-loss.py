import fastf1
import pandas as pd
import numpy as np
import datetime
from fastf1.ergast import Ergast
from os import path
from pit_constants import PIT_LANE_TIME_S, TYPICAL_STATIONARY_TIME_S

DATA_DIR = 'data_files/'
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

ergast = Ergast(result_type='pandas', auto_cast=True)
current_year = datetime.datetime.now().year

pitStops = pd.read_csv(path.join(DATA_DIR, 'f1PitStopsData_Grouped.csv'), sep='\t')

# Add a column with the pit lane time constant for each grandPrixId
pitStops['pit_lane_time_constant'] = pitStops['grandPrixId'].map(PIT_LANE_TIME_S)

# If you want to fill missing values with a default (e.g., np.nan or a specific value)
pitStops['pit_lane_time_constant'] = pitStops['pit_lane_time_constant'].fillna(np.nan)

print(pitStops.head(50))

# Now you can use this column in your calculations