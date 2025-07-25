import fastf1
from fastf1.ergast import Ergast
import pandas as pd
import datetime
import json
from os import path
import os
import streamlit as st
import numpy as np
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_bool_dtype
)
import altair as alt
import time
import numpy as np
#import scipy
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
import seaborn as sns
from xgboost import XGBRegressor, XGBClassifier

DATA_DIR = 'data_files/'


st.set_page_config(
   page_title="Formula 1 Analysis",
   page_icon=path.join(DATA_DIR, 'favicon.png'),
   layout="wide",
   initial_sidebar_state="expanded"
)

def km_to_miles(km):
    return km * 0.621371


# @st.cache_data(show_spinner="Loading latest available F1 practice and qualifying data...")
# def get_latest_gp_data():
#     # --- Load race lookup for grandPrixId ---
#     with open(path.join(DATA_DIR, 'f1db-races.json'), 'r', encoding='utf-8') as f:
#         f1db_races = json.load(f)

#     year_round_to_id = {}
#     id_to_year_round = {}
#     for race in f1db_races:
#         yr = int(race['year'])
#         rnd = int(race['round'])
#         rid = race['id']
#         year_round_to_id[(yr, rnd)] = rid
#         id_to_year_round.setdefault(rid, []).append((yr, rnd))
#     for rid in id_to_year_round:
#         id_to_year_round[rid].sort(reverse=True)

#     ergast = Ergast(result_type='pandas', auto_cast=True)
#     today = datetime.date.today()
#     schedule = ergast.get_race_schedule(season=today.year)
#     date_col = [col for col in schedule.columns if 'date' in col.lower()][0]
#     schedule[date_col] = pd.to_datetime(schedule[date_col])

#     upcoming = schedule[schedule[date_col] >= pd.Timestamp(today)].sort_values(date_col).head(1)
#     if upcoming.empty:
#         return None, None, None, None, None

#     race = upcoming.iloc[0]
#     year = int(race['season'])
#     round_num = int(race['round'])
#     grand_prix_id = year_round_to_id.get((year, round_num))
#     if grand_prix_id is None:
#         return None, None, None, None, None

#     # --- 1. Try current year first ---
#     sched = ergast.get_race_schedule(season=year)
#     date_col_sched = [col for col in sched.columns if 'date' in col.lower()][0]
#     sched[date_col_sched] = pd.to_datetime(sched[date_col_sched])
#     race_row = sched[sched['round'].astype(int) == round_num]
#     if not race_row.empty and race_row.iloc[0][date_col_sched].date() <= today:
#         # Try practice
#         for session_type in ['FP3', 'FP2', 'FP1']:
#             try:
#                 session = fastf1.get_session(year, round_num, session_type)
#                 session.load()
#                 if not session.laps.empty:
#                     practice_laps = session.laps.copy()
#                     practice_laps['Session'] = session_type
#                     practice_laps['grandPrixId'] = grand_prix_id
#                     break
#             except Exception:
#                 continue
#         # Try qualifying
#         try:
#             qual_session = fastf1.get_session(year, round_num, 'Q')
#             qual_session.load()
#             if not qual_session.laps.empty:
#                 qualifying_laps = qual_session.laps.copy()
#                 qualifying_laps['Session'] = 'Q'
#                 qualifying_laps['grandPrixId'] = grand_prix_id
#                 st.write(f"Loaded qualifying session for {grand_prix_id} ({year}, Round {round_num})")
#             else:
#                 qualifying_laps = None
#                 st.write(f"No qualifying data available for {grand_prix_id} ({year}, Round {round_num})")
#         except Exception:
#             qualifying_laps = None
#             st.write("Exception loading qualifying session: No qualifying data available for the target")
#         # If either found, return immediately
#         if ('practice_laps' in locals() and practice_laps is not None and not practice_laps.empty) or \
#            ('qualifying_laps' in locals() and qualifying_laps is not None and not qualifying_laps.empty):
#             return practice_laps, qualifying_laps, grand_prix_id, year, round_num

#     # --- 2. Fallback: Try previous years ---
#     practice_laps = None
#     qualifying_laps = None
#     used_year = None
#     used_round = None

#     today = datetime.date.today()
#     for yr, rnd in id_to_year_round[grand_prix_id]:
#         # Get the race date for this year/round
#         sched = ergast.get_race_schedule(season=yr)
#         date_col_sched = [col for col in sched.columns if 'date' in col.lower()][0]
#         sched[date_col_sched] = pd.to_datetime(sched[date_col_sched])
#         race_row = sched[sched['round'].astype(int) == rnd]
#         if race_row.empty:
#             continue
#         race_date = race_row.iloc[0][date_col_sched].date()
#         if race_date > today:
#             continue  # Skip future races

#         st.write(f"Trying year={yr}, round={rnd} for grandPrixId={grand_prix_id} (race date: {race_date})")

#         # Try practice
#         for session_type in ['FP3', 'FP2', 'FP1']:
#             try:
#                 session = fastf1.get_session(yr, rnd, session_type)
#                 session.load()
#                 if not session.laps.empty:
#                     practice_laps = session.laps.copy()
#                     practice_laps['Session'] = session_type
#                     practice_laps['grandPrixId'] = grand_prix_id
#                     break
#             except Exception as e:
#                 print(f"{session_type} not available for year={yr}, round={rnd}: {e}")
#         # Try qualifying
#         try:
#             qual_session = fastf1.get_session(yr, rnd, 'Q')
#             qual_session.load()
#             if not qual_session.laps.empty:
#                 qualifying_laps = qual_session.laps.copy()
#                 qualifying_laps['Session'] = 'Q'
#                 qualifying_laps['grandPrixId'] = grand_prix_id
#         except Exception as e:
#             print(f"Qualifying not available for year={yr}, round={rnd}: {e}")
#         # If either found, break
#         if (practice_laps is not None and not practice_laps.empty) or (qualifying_laps is not None and not qualifying_laps.empty):
#             used_year = yr
#             used_round = rnd
#             st.write(f"Using data from year={yr}, round={rnd}, grandPrixId={grand_prix_id}")
#             break

#     return practice_laps, qualifying_laps, grand_prix_id, used_year, used_round

# Usage in your Streamlit app:
#practice_laps, qualifying_laps, grand_prix_id, used_year, used_round = get_latest_gp_data()


# --- Pull most recent available practice session (FP3, FP2, FP1) ---
#practice_laps = None
#for session_type in ['FP3', 'FP2', 'FP1']:
#    try:
#        session = fastf1.get_session(year, round_num, session_type)
#        session.load()
#        if not session.laps.empty:
#            practice_laps = session.laps.copy()
#            practice_laps['Session'] = session_type
#            practice_laps['grandPrixId'] = grand_prix_id
#            print(f"Loaded {session_type} for grandPrixId={grand_prix_id} ({year})")
#            break
#    except Exception as e:
#        print(f"{session_type} not available for grandPrixId={grand_prix_id} ({year}): {e}")

# --- Pull qualifying session ---
#try:
#    qual_session = fastf1.get_session(year, round_num, 'Q')
#    qual_session.load()
#    if not qual_session.laps.empty:
#        qualifying_laps = qual_session.laps.copy()
#        qualifying_laps['Session'] = 'Q'
#        qualifying_laps['grandPrixId'] = grand_prix_id
#        print("Loaded qualifying session.")
#    else:
#        qualifying_laps = None
#        print("No qualifying data available for the target race.")
#except Exception as e:
#    print(f"Could not load qualifying session: {e}")
#    qualifying_laps = None

# Now you have:
# - practice_laps: DataFrame for the most recent available practice session for the upcoming event (by grandPrixId)
# - qualifying_laps: DataFrame for the most recent available qualifying session for the upcoming event (by grandPrixId)


# Load race lookup for grandPrixId
# with open(path.join(DATA_DIR, 'f1db-races.json'), 'r', encoding='utf-8') as f:
#     f1db_races = json.load(f)

# ergast = Ergast(result_type='pandas', auto_cast=True)
# today = datetime.date.today()
# schedule = ergast.get_race_schedule(season=today.year)
# date_col = [col for col in schedule.columns if 'date' in col.lower()][0]
# schedule[date_col] = pd.to_datetime(schedule[date_col])

# # Find the next race (date in the future)
# upcoming = schedule[schedule[date_col] >= pd.Timestamp(today)].sort_values(date_col).head(1)
# if upcoming.empty:
#     raise Exception("No upcoming race found in the schedule.")

# race = upcoming.iloc[0]
# event_name = race['raceName']
# target_year = int(race['season'])

# # Try to get this year's round and id, else fallback to previous years
# # Build lookups
# year_round_to_id = {}
# id_to_year_round = {}
# for race in f1db_races:
#     yr = int(race['year'])
#     rnd = int(race['round'])
#     rid = race['id']
#     year_round_to_id[(yr, rnd)] = rid
#     id_to_year_round.setdefault(rid, []).append((yr, rnd))
# for rid in id_to_year_round:
#     id_to_year_round[rid].sort(reverse=True)  # Most recent first

# # Get next race's year and round
# race = upcoming.iloc[0]
# year = int(race['season'])
# round_num = int(race['round'])

# # Get grandPrixId for the upcoming race
# grand_prix_id = year_round_to_id.get((year, round_num))
# if grand_prix_id is None:
#     raise Exception(f"No grandPrixId found for year={year}, round={round_num}")

# # Find the most recent (year, round) for this grandPrixId, not after the current year
# candidates = [yr_rnd for yr_rnd in id_to_year_round[grand_prix_id] if yr_rnd[0] <= year]
# if not candidates:
#     raise Exception(f"No previous races found for grandPrixId={grand_prix_id}")
# use_year, use_round = candidates[0]
# print(f"Using data from year={use_year}, round={use_round}, grandPrixId={grand_prix_id}")

# print(f"Target race: {event_name} ({year} Round {round_num}), grandPrixId: {grand_prix_id}")

# # --- Pull most recent available practice session (FP3, FP2, FP1) ---
# practice_laps = None
# for session_type in ['FP3', 'FP2', 'FP1']:
#     try:
#         session = fastf1.get_session(year, round_num, session_type)
#         session.load()
#         if not session.laps.empty:
#             practice_laps = session.laps.copy()
#             practice_laps['Session'] = session_type
#             practice_laps['grandPrixId'] = grand_prix_id
#             print(f"Loaded {session_type} for {event_name} ({year})")
#             break
#     except Exception as e:
#         print(f"{session_type} not available for {event_name} ({year}): {e}")

# # --- Pull qualifying session ---
# try:
#     qual_session = fastf1.get_session(year, round_num, 'Q')
#     qual_session.load()
#     if not qual_session.laps.empty:
#         qualifying_laps = qual_session.laps.copy()
#         qualifying_laps['Session'] = 'Q'
#         qualifying_laps['grandPrixId'] = grand_prix_id
#         print("Loaded qualifying session.")
#     else:
#         qualifying_laps = None
#         print("No qualifying data available for the target race.")
# except Exception as e:
#     print(f"Could not load qualifying session: {e}")
#     qualifying_laps = None

# Now you have:
# - practice_laps: DataFrame for the most recent practice session (FP3/FP2/FP1) for the upcoming event (this year or fallback)
# - qualifying_laps: DataFrame for the qualifying session for the upcoming event (this year or fallback)
# Both include grandPrixId for cross-referencing

#variable_to_change = "helloWorld123"
#variable_changed = re.sub( r"([A-Z])|([0-9]+)", r" \1\2", variable_to_change).strip()

#print(f"Updated Variable name {variable_changed.title()}")

##### to do: modify variable names for display

def get_last_modified_file(dir_path):
    try:
        files = [path.join(dir_path, f) for f in os.listdir(dir_path) if path.isfile(os.path.join(dir_path, f))]
        if not files:
            return None
        
        last_modified_file = max(files, key=path.getmtime)
        return last_modified_file
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return None

latest_file = get_last_modified_file(DATA_DIR)
modification_time = path.getmtime(latest_file)
#readable_time = time.ctime(modification_time)
readable_time = datetime.datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d %I:%M %p')

def reset_filters():
    # Assuming you have filters stored in session state
    print(f"Session keys: {st.session_state.keys()}")
    for key in st.session_state.keys():
        if key.startswith('filter_'):
            st.session_state[key] = None


def highlight_correlation(val):

    if val >= 0.6 and val < 1.0:            
        color = 'green'
    elif val < -0.6:
        color = 'red' 
    else:
        color = 'white'
    return f'background-color: {color}'

#def reset():
#    st.session_state.selection = ' All'

#st.sidebar.button('Reset', on_click=reset_filters())

column_rename_for_filter = {
    'constructorName': 'Constructor', 
    'grandPrixName': 'Race',
    'grandPrixYear': 'Year',
    'positionsGained': 'Positions Gained',
    'resultsDriverName': 'Driver',
    'resultsFinalPositionNumber' : 'Final Position',
    'resultsPodium': 'Podium',
    'resultsStartingGridPositionNumber': 'Starting Position',
    'resultsTop10': 'Top 10',
    'resultsTop5': 'Top 5',
    'short_date': 'Race Date',
    'DNF' : 'DNF', 
    'resultsReasonRetired': 'Reason Retired',
    'averagePracticePosition': 'Average Practice Pos.', 
    'lastFPPositionNumber': 'Last Free Practice Pos.', 
    'resultsQualificationPositionNumber': 'Qualifying Pos.', 
    'q1End': 'Out at Q1', 
    'q2End': 'Out at Q2', 
    'q3Top10': 'Q3 Top 10',
    'numberOfStops': 'Number of Stops',
    'averageStopTime': 'Average Pit Stop Time (s)',
    'totalStopTime': 'Total Pit Stop Time (s)',
    'grandPrixLaps': 'Laps (Race)', 
    'constructorTotalRaceStarts': 'Constructor Total Starts',
    'constructorTotalRaceWins': 'Constructor Total Wins',
    'constructorTotalPolePositions': 'Total Pole Positions',
    'turns': 'Turns (Race)',
    'driverBestStartingGridPosition': 'Best Starting Grid Position (Driver)',
    'driverBestRaceResult': 'Best Result (Driver)',
    'driverTotalChampionshipWins': 'Total Championship Wins (Driver)',
    'driverTotalRaceEntries': 'Total Entries (Driver)', 
    'driverTotalRaceStarts': 'Total Starts (Driver)',
    'driverTotalRaceWins': 'Total Wins (Driver)', 
    'driverTotalRaceLaps': 'Total Laps (Driver)', 
    'driverTotalPodiums': 'Total Podiums (Driver)',
    'driverTotalPolePositions': 'Total Pole Positions (Driver)',
    'activeDriver': 'Active Driver (Raced this year)',
    'yearsActive': 'Years Active',
    'streetRace' : 'Street',
    'trackRace': 'Track', 
    'Points': 'Points (Driver)',
    'constructorRank': 'Constructor Rank',
    'driverRank': 'Driver Rank',
    'bestChampionshipPosition': 'Best Champ Pos.',
    'bestStartingGridPosition': 'Best Starting Grid Pos.',
    'bestRaceResult': 'Best Race Result',
    'totalChampionshipWins': 'Total Champ Wins',
    'totalRaceEntries': 'Total Race Entries',
    'totalRaceStarts': 'Total Race Starts',
    'totalRaceWins': 'Total Race Wins',
    'total1And2Finishes': 'Total 1st and 2nd',
    'totalRaceLaps': 'Total Race Laps (Construtor)',
    'totalPodiums': 'Total Podiums (Construtor)',
    'totalPodiumRaces': 'Total Podium Races (Construtor)',
    'totalPoints' : 'Total Points',
    'totalChampionshipPoints': 'Total Champ Points',
    'totalPolePositions' : 'Total Pole Positions',
    'totalFastestLaps': 'Total Fastest Laps',
    'bestQualifyingTime_sec': 'Best Qualifying Time (s)',
    }         

individual_race_grouped_columns_to_display = {
    'resultsDriverName': st.column_config.TextColumn("Driver"),
    'constructorName': st.column_config.TextColumn("Constructor"),
    'average_starting_position': st.column_config.NumberColumn("Avg Starting Pos.", format="%.2f"),
    'average_ending_position': st.column_config.NumberColumn("Avg Final Pos.", format="%.2f"),
    'average_positions_gained': st.column_config.NumberColumn("Avg Positions Gained", format="%.2f"),
    'driver_races': st.column_config.NumberColumn("# of Races", format="%d")
}

flags_safety_cars_columns_to_display = {
    'grandPrixYear': st.column_config.NumberColumn("Year", format="%d"),
    'round': st.column_config.NumberColumn("Round", format="%d"),
    'raceId': None,
    'grandPrixId': None,
    'SafetyCarStatus': st.column_config.NumberColumn("Safety Car", format="%d"),
    'redFlag': st.column_config.NumberColumn("Red Flag"),
    'yellowFlag': st.column_config.NumberColumn("Yellow Flag"),
    'doubleYellowFlag': st.column_config.NumberColumn("Double Yellow Flag"),
    'dnf_count': st.column_config.NumberColumn("DNF Count", format="%d")

}

predicted_position_columns_to_display = {
    'resultsDriverName': st.column_config.TextColumn("Driver"),
    'constructorName': st.column_config.TextColumn("Constructor"),
    'resultsStartingGridPositionNumber': st.column_config.NumberColumn(
        "Starting Grid Position", format="%d", min_value=1, max_value=20, step=1, default=1),
    'resultsFinalPositionNumber': st.column_config.NumberColumn(
        "Final Position", format="%d", min_value=1, max_value=20, step=1, default=1),
    'PredictedFinalPosition': st.column_config.NumberColumn("Predicted Final Position", format="%.3f"),
    'grandPrixName': None, 'totalChampionshipPoints': None, 'driverTotalChampionshipWins': None,
    'resultsStartingGridPositionNumber': None, 'averagePracticePosition': None, 'totalFastestLaps': None, 'total1And2Finishes': None,
    'lastFPPositionNumber': None, 'resultsQualificationPositionNumber': None, 'constructorTotalRaceStarts': None, 
    'constructorTotalRaceWins': None, 'constructorTotalPolePositions': None, 'driverTotalRaceEntries': None, 
    'totalPolePositions': None, 'Points': None, 'driverTotalRaceStarts': None, 'driverTotalRaceWins': None, 
    'driverTotalPodiums': None, 'driverRank': None, 'constructorRank': None, 'driverTotalPolePositions': None, 
    'yearsActive': None, 'bestQualifyingTime_sec': None, 'resultsDriverId': None
}
predicted_dnf_position_columns_to_display = {
    'resultsDriverName': st.column_config.TextColumn("Driver"),
    'constructorName': st.column_config.TextColumn("Constructor"),
    'resultsStartingGridPositionNumber': st.column_config.NumberColumn(
        "Starting Grid Position", format="%d", min_value=1, max_value=20, step=1, default=1),
    'resultsFinalPositionNumber': st.column_config.NumberColumn(
        "Final Position", format="%d", min_value=1, max_value=20, step=1, default=1),
    'PredictedDNFProbability': st.column_config.NumberColumn("Predicted DNF Probability", format="%.3f"),
    'PredictedDNFProbabilityPercentage': st.column_config.NumberColumn("Predicted DNF (%)", format="%.3f"),
    'driverDNFCount': st.column_config.NumberColumn("DNF Count", format="%d"),
    'driverDNFPercentage': st.column_config.NumberColumn("DNF (%)", format="%.3f"),
    'grandPrixName': None, 'totalChampionshipPoints': None, 'driverTotalChampionshipWins': None,
    'resultsStartingGridPositionNumber': None, 'averagePracticePosition': None, 'totalFastestLaps': None, 'total1And2Finishes': None,
    'lastFPPositionNumber': None, 'resultsQualificationPositionNumber': None, 'constructorTotalRaceStarts': None, 
    'constructorTotalRaceWins': None, 'constructorTotalPolePositions': None, 'driverTotalRaceEntries': None, 
    'totalPolePositions': None, 'Points': None, 'driverTotalRaceStarts': None, 'driverTotalRaceWins': None, 
    'driverTotalPodiums': None, 'driverRank': None, 'constructorRank': None, 'driverTotalPolePositions': None, 
    'yearsActive': None, 'bestQualifyingTime_sec': None, 'resultsDriverId': None
}


current_year = datetime.datetime.now().year
raceNoEarlierThan = current_year - 10


@st.cache_data
def load_correlation(nrows):
    correlation_matrix = pd.read_csv(path.join(DATA_DIR, 'f1PositionCorrelation.csv'), sep='\t', nrows=nrows)

    return correlation_matrix
correlation_matrix = load_correlation(10000)

@st.cache_data
def load_data_schedule(nrows):
    raceSchedule = pd.read_json(path.join(DATA_DIR, 'f1db-races.json'))
    grandPrix = pd.read_json(path.join(DATA_DIR, 'f1db-grands-prix.json'))
    raceSchedule = raceSchedule.merge(grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_grandPrix', '_schedule'])
    #raceSchedule = raceSchedule.merge(grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_grandPrix', '_schedule'])
    return raceSchedule

raceSchedule = load_data_schedule(10000)

@st.cache_data
def load_drivers(nrows):
    drivers = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json'))
    return drivers

drivers = load_drivers(10000)

@st.cache_data
def load_qualifying(nrows):
    qualifying = pd.read_csv(path.join(DATA_DIR, 'all_qualifying_races.csv'), sep='\t')
    return qualifying

qualifying = load_qualifying(10000)

@st.cache_data
def load_practices(nrows):
    practices = pd.read_csv(path.join(DATA_DIR, 'all_practice_laps.csv'), sep='\t', dtype={'PitOutTime': str}) 
    return practices

practices = load_practices(10000)

@st.cache_data
def load_data_race_messages(nrows):
    race_messages = pd.read_csv(path.join(DATA_DIR, 'race_control_messages_grouped_with_dnf.csv'),sep='\t')

    return race_messages

race_messages = load_data_race_messages(10000)


schedule_columns_to_display = {
    'round': st.column_config.NumberColumn("Round", format="%d"),
    'fullName': st.column_config.TextColumn("Name"),
    'date': st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
    'time': st.column_config.TimeColumn("Time", format="localized"),
    'courseLength': st.column_config.NumberColumn("Lap Length (km)", format="%.3f"),
    'laps': st.column_config.NumberColumn("Number of Laps", format="%d"),
    'turns': st.column_config.NumberColumn("Number of Turns", format="%d"),
    'distance': st.column_config.NumberColumn("Distance (km)", format="%.3f"),
    'totalRacesHeld': st.column_config.NumberColumn("Races Held", format="%d"),
    'circuitType': st.column_config.TextColumn("Type"),
    'year': None,
    'id_grandPrix': None,
    'grandPrixId': None,
    'qualifyingFormat': None,
    'circuitId': None,
    'direction': None, 
    'countryId': None,
    'abbreviation': None,
    'shortName': None,
    'id_schedule': None,
    'warmingUpTime': None,
    'warmingUpDate': None,
    'sprintRaceTime': None,
    'officialName': None,
    'sprintQualifyingFormat': None,
    'scheduledLaps': None,
    'scheduledDistance': None,
    'driversChampionshipDecider': None,
    'constructorsChampionshipDecider': None,
    'preQualifyingDate': None,
    'preQualifyingTime': None,
    'freePractice1Date': None,
    'freePractice2Date': None,
    'freePractice3Date': None,
    'freePractice4Date': None,
    'freePractice1Time': None,
    'freePractice2Time': None,
    'freePractice3Time': None,
    'freePractice4Time': None,
    'qualifying1Date': None,
    'qualifying2Date': None,
    'qualifying3Date': None,
    'qualifying1Time': None,
    'qualifying2Time': None,
    'qualifying3Time': None,
    'name': None,
    'qualifyingDate': None,
    'qualifyingTime': None,
    'sprintRaceDate': None,
    'sprintQualifyingDate': None,
    'sprintQualifyingTime': None,   
}

@st.cache_data
def load_weather_data(nrows):
    weather = pd.read_csv(path.join(DATA_DIR, 'f1WeatherData_Grouped.csv'), sep='\t', nrows=nrows, usecols=['grandPrixId', 'short_date', 'average_temp', 'total_precipitation', 'average_humidity', 'average_wind_speed', 'id_races'])
    grandPrix = pd.read_json(path.join(DATA_DIR, 'f1db-grands-prix.json'))
    weather_with_grandprix = pd.merge(weather, grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_weather', '_grandPrix'])
    return weather_with_grandprix

weatherData = load_weather_data(10000)


weather_columns_to_display = {
    'short_date': st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
    'average_temp': st.column_config.NumberColumn("Average Temperature (F)", format="%.2f"),
    'total_precipitation': st.column_config.NumberColumn("Precipitation (in)", format="%.2f"),
    'average_humidity': st.column_config.NumberColumn("Average Humidity (%)", format="%.2f"),
    'average_wind_speed': st.column_config.NumberColumn("Average Wind Speed (mph)", format="%.2f"),
    'grandPrixId': None,
    'countryId': None,
    'abbreviation': None,
    'shortName': None,
    'name' : None,
    'fullName': None,
    'id': None, 
    'totalRacesHeld' : None,
    'id_races': None,
}

st.image(path.join(DATA_DIR, 'formula1_logo.png'))
st.title(f'F1 Races from {raceNoEarlierThan} to {current_year}')
st.caption(f"Last updated: {readable_time}")

columns_to_display = {'grandPrixYear': st.column_config.NumberColumn("Year", format="%d"),
    'grandPrixName': st.column_config.TextColumn("Grand Prix"),
    'constructorName': st.column_config.TextColumn("Constructor"),
    'resultsReasonRetired': st.column_config.TextColumn("Reason Retired"),
    'resultsDriverName': st.column_config.TextColumn("Driver"),
    'resultsPodium': st.column_config.CheckboxColumn("Podium"),
    'resultsTop5': st.column_config.CheckboxColumn("Top 5"),
    'resultsTop10': st.column_config.CheckboxColumn("Top 10"),
    'resultsStartingGridPositionNumber': st.column_config.NumberColumn(
        "Starting Grid Position", format="%d", min_value=1, max_value=20, step=1, default=1),
    'resultsFinalPositionNumber': st.column_config.NumberColumn(
        "Final Position", format="%d", min_value=1, max_value=20, step=1, default=1),
    'positionsGained': st.column_config.NumberColumn(
        "Positions Gained", format="%d", min_value=-10, max_value=10, step=1, default=0),
    'activeDriver': None,
    'short_date': None,
    'raceId_results': None,
    'grandPrixRaceId': None,
    'bestQualifyingTime_sec': st.column_config.NumberColumn("Best Qualifying Time (s)", format="%.3f"),
    'DNF': st.column_config.CheckboxColumn("DNF"),
    'streetRace': st.column_config.CheckboxColumn("Street Race"),
    'trackRace': st.column_config.CheckboxColumn("Track Race"),
    'averagePracticePosition': st.column_config.NumberColumn(
        "Avg Practice Pos.", format="%d", min_value=1, max_value=20, step=1, default=1),
    'lastFPPositionNumber': st.column_config.NumberColumn(
        "Last FP Pos.", format="%d", min_value=1, max_value=20, step=1, default=1),
    'resultsQualificationPositionNumber': st.column_config.NumberColumn(
        "Qual. Pos.", format="%d", min_value=1, max_value=20, step=1, default=1),
    'q1End': st.column_config.CheckboxColumn("Out at Q1"),
    'q2End': st.column_config.CheckboxColumn("Out at Q2"),
    'q3Top10': st.column_config.CheckboxColumn("Q3 Top 10"),
    'numberOfStops': st.column_config.NumberColumn(
        "Number of Stops", format="%d", min_value=0, max_value=5, step=1, default=0),
    'averageStopTime': st.column_config.NumberColumn(
        "Avg Stop Time (s)", format="%d", min_value=0, max_value=20, step=1, default=0),
    'totalStopTime': st.column_config.NumberColumn(
        "Total Stop Time (s)", format="%d", min_value=0, max_value=100, step=1, default=0),
    'activeDriver' : st.column_config.CheckboxColumn("Active"), 
    'driverId': None,
    'constructorId': None,
    'raceId': None,
    'constructorId_results': None,
    'driverId_results': None,
    'id': None,
    'name': None,
    'fullName': None,
    'countryId': None,
    'bestChampionshipPosition': st.column_config.NumberColumn(
        "Best Championship Pos.", format="%d", min_value=0, max_value=20, step=1, default=0),
    'bestStartingGridPosition': st.column_config.NumberColumn(
        "Best Starting Grid Pos.", format="%d", min_value=0, max_value=20, step=1, default=0),  
    'bestRaceResult': st.column_config.NumberColumn(
        "Best Race Result", format="%d", min_value=0, max_value=20, step=1, default=0),    
    'bestChampionshipPosition': st.column_config.NumberColumn(
        "Best Championship Pos.", format="%d", min_value=0, max_value=20, step=1, default=0), 
    'totalChampionshipWins': st.column_config.NumberColumn(
        "Total Championship Wins", format="%d", min_value=0, max_value=20, step=1, default=0), 
    'totalRaceStarts': st.column_config.NumberColumn(
        "Total Starts", format="%d", min_value=0, max_value=20, step=1, default=0),   
    'totalRaceEntries': st.column_config.NumberColumn(
        "Total Entries", format="%d", min_value=0, max_value=20, step=1, default=0),       
    'totalRaceWins': st.column_config.NumberColumn(
        "Total Wins", format="%d", min_value=0, max_value=20, step=1, default=0),             
    'resultsDriverId': None,
    'grandPrixLaps': st.column_config.NumberColumn(
        "Laps", format="%d", min_value=0, max_value=100, step=1, default=0),
    'constructorTotalRaceStarts': st.column_config.NumberColumn(
        "Constructor Total Starts", format="%d", min_value=0, max_value=100, step=1, default=0),
    'constructorTotalRaceWins': st.column_config.NumberColumn(
        "Constructor Total Wins", format="%d", min_value=0, max_value=100, step=1, default=0), 
    'constructorTotalPolePositions': st.column_config.NumberColumn(
        "Constructor Total Pole Pos.", format="%d", min_value=0, max_value=100, step=1, default=0),
    'turns': st.column_config.NumberColumn(
        "Turns", format="%d", min_value=0, max_value=100, step=1, default=0),
    'driverBestStartingGridPosition': st.column_config.NumberColumn(
        "Best Starting Grid Pos.", format="%d", min_value=0, max_value=100, step=1, default=0),
    'total1And2Finishes': st.column_config.NumberColumn(
        "Total 1st and 2nd", format="%d", min_value=0, max_value=20, step=1, default=0),     
    'totalRaceLaps': st.column_config.NumberColumn(
        "Total Laps", format="%d", min_value=0, max_value=20, step=1, default=0),     
    'totalPodiums': st.column_config.NumberColumn(
        "Total Podiums", format="%d", min_value=0, max_value=20, step=1, default=0),  
    'totalPodiumRaces': st.column_config.NumberColumn(
        "Total Podium Races", format="%d", min_value=0, max_value=20, step=1, default=0),   
    'totalPoints': st.column_config.NumberColumn(
        "Total Points", format="%d", min_value=0, max_value=20, step=1, default=0),         
    'totalChampionshipPoints': st.column_config.NumberColumn(
        "Total Champ. Points", format="%d", min_value=0, max_value=20, step=1, default=0),    
    'totalPolePositions': st.column_config.NumberColumn(
        "Total Pole Pos.", format="%d", min_value=0, max_value=20, step=1, default=0), 
    'totalFastestLaps': st.column_config.NumberColumn(
        "Total Fastest Laps", format="%d", min_value=0, max_value=20, step=1, default=0),  
    'TeamName': None,
    'driverId_driver_standings': None,        
    'constructorRank': st.column_config.NumberColumn(
        "Constructor Rank", format="%d", min_value=0, max_value=20, step=1, default=0),    
    'driverName': None,
    'points': st.column_config.NumberColumn(
        "Current Year Points", format="%d", min_value=0, max_value=20, step=1, default=0),
    'driverRank': st.column_config.NumberColumn(
        "Current Year Rank", format="%d", min_value=0, max_value=20, step=1, default=0),                                      
    'driverBestRaceResult': st.column_config.NumberColumn("Best Result", format="%d", min_value=0, max_value=100, step=1, default=0),
    'driverTotalChampionshipWins': st.column_config.NumberColumn(
        "Total Championship Wins", format="%d", min_value=0, max_value=100, step=1, default=0),
    'driverTotalRaceEntries': st.column_config.NumberColumn("Total Race Entries", format="%d", min_value=0, max_value=100, step=1, default=0),   
    'driverTotalRaceStarts': st.column_config.NumberColumn("Total Race Starts", format="%d", min_value=0, max_value=100, step=1, default=0),   
    'driverTotalRaceWins': st.column_config.NumberColumn("Total Wins", format="%d", min_value=0, max_value=100, step=1, default=0),   
    'driverTotalRaceLaps': st.column_config.NumberColumn("Total Laps", format="%d", min_value=0, max_value=100, step=1, default=0),   
    'driverTotalPodiums': st.column_config.NumberColumn("Total Podiums", format="%d", min_value=0, max_value=100, step=1, default=0),
    'driverTotalPolePositions': st.column_config.NumberColumn("Total Pole Positions", format="%d", min_value=0, max_value=100, step=1, default=0),
    'yearsActive': st.column_config.NumberColumn(
        "Years Active", format="%d", min_value=0, max_value=100, step=1, default=0),

}

correlation_columns_to_display = {
    'Unnamed: 0': st.column_config.TextColumn("Field"),
    'resultsPodium': st.column_config.NumberColumn("Podium", format="%.3f"),
    'resultsTop5': st.column_config.NumberColumn("Top 5", format="%.3f"),
    'resultsTop10': st.column_config.NumberColumn("Top 10", format="%.3f"),
    'resultsStartingGridPositionNumber': st.column_config.NumberColumn(
        "Starting Grid Position", format="%.3f"),
    'resultsFinalPositionNumber': st.column_config.NumberColumn(
        "Final Position", format="%.3f"),
    'positionsGained': st.column_config.NumberColumn(
        "Positions Gained", format="%.3f"),
    'DNF': st.column_config.NumberColumn("DNF", format="%.3f"),
    'averagePracticePosition': st.column_config.NumberColumn(
        "Avg Practice Pos.", format="%.3f"),
    'grandPrixLaps': st.column_config.NumberColumn(
        "Laps", format="%.3f"),
    'lastFPPositionNumber': st.column_config.NumberColumn(
        "Last FP Pos.", format="%.3f"),
    'resultsQualificationPositionNumber': st.column_config.NumberColumn(
        "Qual. Pos.", format="%.3f"),
    'constructorTotalRaceStarts': st.column_config.NumberColumn(
        "Constructor Race Starts", format="%.3f"),
    'constructorTotalRaceWins': st.column_config.NumberColumn(
        "Constructor Race Wins", format="%.3f"),
    'constructorTotalPolePositions': st.column_config.NumberColumn(
        "Constructor Pole Pos.", format="%.3f"),
    'turns': st.column_config.NumberColumn("Turns", format="%.3f"),    
    'q1End': st.column_config.NumberColumn("Out at Q1", format="%.3f"),
    'q2End': st.column_config.NumberColumn("Out at Q2", format="%.3f"),
    'q3Top10': st.column_config.NumberColumn("Q3 Top 10", format="%.3f"),
    'numberOfStops': st.column_config.NumberColumn("Number of Stops", format="%.3f"),
    'driverBestStartingGridPosition': st.column_config.NumberColumn(
        "Best Starting Grid Pos.", format="%.3f"),
    'driverBestRaceResult': st.column_config.NumberColumn("Best Result", format="%.3f"),
    'driverTotalChampionshipWins': st.column_config.NumberColumn(
        "Total Championship Wins", format="%.3f"),
    'yearsActive': st.column_config.NumberColumn(
        "Years Active", format="%.3f"),    
    'driverTotalRaceEntries': st.column_config.NumberColumn("Total Race Entries", format="%.3f"),   
    'driverTotalRaceStarts': st.column_config.NumberColumn("Total Race Starts", format="%.3f"),   
    'driverTotalRaceWins': st.column_config.NumberColumn("Total Wins", format="%.3f"),   
    'driverTotalRaceLaps': st.column_config.NumberColumn("Total Laps", format="%.3f"),   
    'driverTotalPodiums': st.column_config.NumberColumn("Total Podiums", format="%.3f"),
    'driverTotalPolePositions': st.column_config.NumberColumn("Total Pole Positions", format="%.3f"),
    'streetRace': st.column_config.NumberColumn("Street Race", format="%.3f"), 
    'trackRace': st.column_config.NumberColumn("Track Race", format="%.3f")
}

next_race_columns_to_display = {
    'date': st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
    'time': st.column_config.TimeColumn("Time", format="localized"),
    'fullName': st.column_config.TextColumn("Grand Prix"),
    'courseLength': st.column_config.TextColumn("Lap Length (km)"),
    'turns': st.column_config.TextColumn("Number of Turns"),
    'laps': st.column_config.TextColumn("Number of Laps")    

}

driver_vs_constructor_columns_to_display = {
    'constructorName': st.column_config.TextColumn("Constructor"),
    'resultsDriverName': st.column_config.TextColumn("Driver"),
    'positionsGained': st.column_config.NumberColumn("Positions Gained", format="%d"),
    'average_final_position': st.column_config.NumberColumn("Avg. Final Position", format="%.2f")
}

season_summary_columns_to_display = {
    'resultsDriverName': st.column_config.TextColumn("Driver"),
    'positions_gained': st.column_config.NumberColumn("Positions Gained", format="%d"),
    'total_podiums': st.column_config.NumberColumn("Total Podiums", format="%d")
}

@st.cache_data
def load_data(nrows):
    fullResults = pd.read_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), sep='\t', nrows=nrows, usecols=['grandPrixYear', 'grandPrixName', 'resultsDriverName', 'resultsPodium', 'resultsTop5', 'resultsTop10', 'constructorName',  'resultsStartingGridPositionNumber', 'resultsFinalPositionNumber', 
    'positionsGained', 'short_date', 'raceId_results', 'grandPrixRaceId', 'DNF', 'averagePracticePosition', 'lastFPPositionNumber', 'resultsQualificationPositionNumber', 'q1End', 'q2End', 'q3Top10', 'resultsDriverId', 
    'grandPrixLaps', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions', 'turns', 'resultsReasonRetired', 'constructorId_results', 
    'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalPolePositions', 'activeDriver', 'streetRace', 'trackRace', #'Points',
           'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 'bestQualifyingTime_sec', 'yearsActive', 'driverDNFCount', 'driverDNFAvg',
           'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec', 'LapTime_sec', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'avgLapPace', 'finishingTime'
           ], dtype={'resultsStartingGridPositionNumber': 'Float64', 'resultsFinalPositionNumber': 'Float64', 'positionsGained': 'Int64', 'averagePracticePosition': 'Float64', 'lastFPPositionNumber': 'Float64', 'resultsQualificationPositionNumber': 'Int64'})
    
    pitStops = pd.read_csv(path.join(DATA_DIR, 'f1PitStopsData_Grouped.csv'), sep='\t', nrows=nrows, usecols=['raceId', 'driverId', 'constructorId', 'numberOfStops', 'averageStopTime', 'totalStopTime'])
    constructor_standings = pd.read_csv(path.join(DATA_DIR, 'constructor_standings.csv'), sep='\t')
    driver_standings = pd.read_csv(path.join(DATA_DIR, 'driver_standings.csv'), sep='\t')

    fullResults = pd.merge(fullResults, pitStops, left_on=['raceId_results', 'resultsDriverId'], right_on=['raceId', 'driverId'], how='left', suffixes=['_results', '_pitStops'])
    fullResults = pd.merge(fullResults, constructor_standings, left_on='constructorId_results', right_on='id', how='left', suffixes=['_results', '_constructor_standings'])
    fullResults = pd.merge(fullResults, driver_standings, left_on='resultsDriverId', right_on='driverId', how='left', suffixes=['_results', '_driver_standings'])
    # Select only the columns you want from weatherData
    weather_fields = ['id_races', 'average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation']  # add more as needed
    weatherData_subset = weatherData[weather_fields]
    fullResults = pd.merge(fullResults, weatherData_subset, left_on='raceId_results', right_on='id_races', how='left', suffixes=['_results', '_weather'])

    # fullResults['raceId_results'] = pd.to_numeric(fullResults['raceId_results'], errors='coerce')
    # qualifying['raceId'] = pd.to_numeric(qualifying['raceId'], errors='coerce')

    # fullResults['resultsDriverId'] = pd.to_numeric(fullResults['resultsDriverId'], errors='coerce')
    # qualifying['driverId'] = pd.to_numeric(qualifying['driverId'], errors='coerce')

    fullResults = pd.merge(fullResults, qualifying, left_on=['raceId_results', 'resultsDriverId'], right_on=['raceId', 'driverId'], how='left', suffixes=['_results_with_qualifying', '_qualifying'])
    fullResults.drop_duplicates(subset=['grandPrixYear', 'grandPrixName', 'resultsDriverName'], inplace=True)
    ####### need to merge practices and qualifying data so that I can use it in the historical model

    #### merge qualifying on raceID as well
    # fullResults.rename(columns={'Points_results_with_qualifying': 'Points',})
    return fullResults

data = load_data(10000)

## Most recent date with weather
#print(data['short_date'].max())

# Round averagePracticePosition to 2 decimal places
data['averagePracticePosition'] = data['averagePracticePosition'].round(2)

# Convert columns to appropriate types to allow for NaN values
data['resultsStartingGridPositionNumber'] = data['resultsStartingGridPositionNumber'].astype('Float64')
data['resultsFinalPositionNumber'] = data['resultsFinalPositionNumber'].astype('Float64')
data['positionsGained'] = data['positionsGained'].astype('Int64')
data['averagePracticePosition'] = data['averagePracticePosition'].astype('Float64')
data['lastFPPositionNumber'] = data['lastFPPositionNumber'].astype('Float64')
data['resultsQualificationPositionNumber'] = data['resultsQualificationPositionNumber'].astype('Int64')
data['short_date'] = pd.to_datetime(data['short_date'])
data['numberOfStops'] = data['numberOfStops'].astype('Int64')
data['averageStopTime'] = data['averageStopTime'].astype('Float64')
data['totalStopTime'] = data['totalStopTime'].astype('Float64')
data['driverBestStartingGridPosition'] = data['driverBestStartingGridPosition'].astype('Int64')
data['driverBestRaceResult'] = data['driverBestRaceResult'].astype('Int64')
data['constructorRank'] = data['constructorRank'].astype('Int64')
data['Points'] = data['Points_results_with_qualifying'].astype('Int64')
data['driverRank'] = data['driverRank'].astype('Int64')
if 'bestQualifyingTime_sec' in data.columns:
    data['bestQualifyingTime_sec'] = data['bestQualifyingTime_sec'].astype('Float64')
else:
    st.warning("'bestQualifyingTime_sec' column not found in data.")
#data['bestQualifyingTime_sec'] = data['bestQualifyingTime_sec'].astype('Float64')
data['driverTotalChampionshipWins'] = data['driverTotalChampionshipWins'].astype('Int64')
data['driverTotalRaceEntries'] = data['driverTotalRaceEntries'].astype('Int64')
data['bestChampionshipPosition'] = data['bestChampionshipPosition_results_with_qualifying'].astype('Int64')
data['bestStartingGridPosition'] = data['bestStartingGridPosition_results_with_qualifying'].astype('Int64')
data['bestRaceResult'] = data['bestRaceResult_results_with_qualifying'].astype('Int64')
data['totalChampionshipWins'] = data['totalChampionshipWins_results_with_qualifying'].astype('Int64')
data['totalRaceStarts'] = data['totalRaceStarts_results_with_qualifying'].astype('Int64')
data['totalRaceWins'] = data['totalRaceWins_results_with_qualifying'].astype('Int64')
data['total1And2Finishes'] = data['total1And2Finishes'].astype('Int64')
data['totalRaceLaps'] = data['totalRaceLaps_results_with_qualifying'].astype('Int64')
data['totalPodiums'] = data['totalPodiums_results_with_qualifying'].astype('Int64')
data['totalPodiumRaces'] = data['totalPodiumRaces'].astype('Int64')
data['totalPoints'] = data['totalPoints_results_with_qualifying'].astype('Float64')
data['totalChampionshipPoints'] = data['totalChampionshipPoints_results_with_qualifying'].astype('Float64')
data['totalPolePositions'] = data['totalPolePositions_results_with_qualifying'].astype('Int64')
data['totalFastestLaps'] = data['totalFastestLaps_results_with_qualifying'].astype('Int64')
data['totalRaceEntries'] = data['totalRaceEntries_results_with_qualifying'].astype('Int64')


column_names = data.columns.tolist()

## do not create filters for any field in this list
## could be to avoid duplicates or not useful for filtering
exclusionList = ['grandPrixRaceId', 'raceId_results',  'constructorId', 'driverId', 'resultsDriverId', 'HeadshotUrl', 'DriverId', 'firstName', 'lastName',
                 'raceId', 'id', 'id_grandPrix', 'id_schedule', 'bestQualifyingTime_sec', 'TeamName', 'circuitId', 'grandPrixRaceId', 'grandPrixId', 'Abbreviation',
                'driverId_driver_standings', 'constructorId_results', 'driverId_results', 'driverId_driver_standings', 'TeamId', 'TeamColor', 'BroadcastName',
                'driverName', 'driverId_driver_standings', 'countryId', 'name', 'fullName', 'points', 'abbreviation', 'shortName', 'id', 'constructorId_results', 'driverId_results',
                'nationalityCountryId', 'secondNationalityCountryId', 'countryOfBirthCountryId', 'placeOfBirth', 'dateOfDeath', 'dateOfBirth', 'gender', 'permanentNumber',
                 'Q1', 'Q2', 'Q3', 'Time', 'PitOutTime', 'PitInTime', 'PitStopTime_sec', 'PitStopTime_mph', 'PitStopTime_mph_avg', 'PitStopTime_sec_avg',
                 'DriverNumber', 'FirstName', 'LastName', 'FullName', 'CountryCode', 'Position', 'ClassifiedPosition', 'GridPosition', 'Status', 
                 'driverNumber', 'Round', 'Year', 'Event', 'totalDriverOfTheDay', 'totalGrandSlams', 'finishingTime']

suffixes_to_exclude = ('_x', '_y', '_qualifying', '_results_with_qualifying', '_drivers', '_mph', '_sec', '.1', '.2', '.3')
auto_exclusions = [col for col in column_names if col.endswith(suffixes_to_exclude)]
exclusionList = exclusionList + auto_exclusions

# If errant/extra columns appear on the left in filters, the below analysis will point them out.

# st.write(f"Exclusion List: {exclusionList}")

# remaining_columns = [col for col in column_names if col not in exclusionList]
# st.write(f"Remaining Columns: {remaining_columns}")

#column_names.sort()

def get_features_and_target(data):
    features = ['grandPrixName', 'constructorName', 'resultsDriverName', 'totalChampionshipPoints', 'driverTotalChampionshipWins',
        'resultsStartingGridPositionNumber', 'averagePracticePosition', 'totalFastestLaps', 'total1And2Finishes',
        'lastFPPositionNumber', 'resultsQualificationPositionNumber', 'constructorTotalRaceStarts', 
        'constructorTotalRaceWins', 'constructorTotalPolePositions', 'driverTotalRaceEntries', 'finishingTime',
        'totalPolePositions', 'Points', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalPodiums', 'driverRank', 'constructorRank', 'driverTotalPolePositions', 
        'yearsActive', 'bestQualifyingTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec', 'LapTime_sec', 
        'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'avgLapPace', 'trackRace', 'streetRace', 'turns',
        'average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation', 'driverDNFCount', 'driverDNFAvg']
    target = 'resultsFinalPositionNumber'
    return data[features], data[target]

def get_preprocessor_position():
    categorical_features = ['grandPrixName', 'constructorName', 'resultsDriverName']
    numerical_features = ['totalChampionshipPoints', 'driverTotalChampionshipWins',
        'resultsStartingGridPositionNumber', 'averagePracticePosition', 'totalFastestLaps', 'total1And2Finishes',
        'lastFPPositionNumber', 'resultsQualificationPositionNumber', 'constructorTotalRaceStarts', 
        'constructorTotalRaceWins', 'constructorTotalPolePositions', 'driverTotalRaceEntries', 'finishingTime',
        'totalPolePositions', 'Points', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalPodiums', 'driverRank', 'constructorRank', 'driverTotalPolePositions', 
        'yearsActive', 'bestQualifyingTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec', 'LapTime_sec', 
        'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'avgLapPace', 'trackRace', 'streetRace', 'turns',
        'average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation', 'driverDNFCount', 'driverDNFAvg']
    
    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', numerical_imputer),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', categorical_imputer),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    return preprocessor

def get_features_and_target_dnf(data):
    features = [
    'grandPrixName', 'constructorName', 'resultsDriverName',
    'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalChampionshipWins',
    'driverTotalRaceWins', 'driverTotalPodiums', 'yearsActive',
    'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions',
    'averagePracticePosition', 'lastFPPositionNumber', 'resultsQualificationPositionNumber',
    'trackRace', 'streetRace', 'turns', 'average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation', 'driverDNFCount', 'driverDNFAvg'
    # Add weather features if available
    ]
    target = 'DNF'
    return data[features], data[target]

def get_preprocessor_dnf():
    categorical_features = ['grandPrixName', 'constructorName', 'resultsDriverName']
    numerical_features = ['driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalChampionshipWins',
    'driverTotalRaceWins', 'driverTotalPodiums', 'yearsActive',
    'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions',
    'averagePracticePosition', 'lastFPPositionNumber', 'resultsQualificationPositionNumber',
    'trackRace', 'streetRace', 'turns', 'average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation', 'driverDNFCount', 'driverDNFAvg']

    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', numerical_imputer),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', categorical_imputer),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    return preprocessor

###### Training model for final racing position prediction

def train_and_evaluate_model(data):
    X, y = get_features_and_target(data)
    preprocessor = get_preprocessor_position()

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'  # or 'auto'
        ))
    ])

    # model = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('regressor', GradientBoostingRegressor(
    #         n_estimators=200,
    #         learning_rate=0.1,
    #         max_depth=4,
    #         random_state=42
    #     ))
    # ])

    if y.isnull().any():
        y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, mse, r2, mae

def train_and_evaluate_dnf_model(data):
    X, y = get_features_and_target_dnf(data)
    preprocessor = get_preprocessor_dnf()
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        ))
    ])
    model.fit(X, y)
    return model

@st.cache_resource
def get_trained_model():
    model, mse, r2, mae = train_and_evaluate_model(data)
    return model

model = get_trained_model()
dnf_model = train_and_evaluate_dnf_model(data)

if st.checkbox('Filter Results'):
    # Create a dictionary to store selected filters for multiple columns
    filters = {}
    #print(column_names)

    filters_for_reset = {}

    # Iterate over the columns to display and create a filter for each
    st.sidebar.header("Select filters to apply:")
    for column in column_names:
        
        for old_column, new_column in column_rename_for_filter.items():
            if column == old_column: 
                column_friendly_name = new_column
        
        if is_numeric_dtype(data[column]) and (data[column].dtype in ('np.int64', 'np.float64', 'Int64', 'int64', 'Float64') ):

            # Do not display if the column is in the exclusion list noted at the top of the file
            if column not in exclusionList:
                min_val, max_val = int(data[column].min()), int(data[column].max())           
                                
                selected_range = st.sidebar.slider(
                    column_friendly_name,
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=1,
                    key=f"range_filter_{column}"
            )
                filters_for_reset[column] = {
                    'key': f"range_filter_{column}",
                    'column': column,
                    'dtype': data[column].dtype,
                    'min': min_val,
                    'max': max_val,
                    'selected_range': selected_range
                }
                #filters_for_reset['type'] = data[column].dtype
                #filters_for_reset['min'] = min_val
                #filters_for_reset['max'] = max_val
                #filters_for_reset['selected_range'] = selected_range

                filters[column] = selected_range
        elif is_datetime64_any_dtype(data[column]):
            # Allow range selection for datetime columns
            min_val, max_val = data[column].min(), data[column].max()
            
            formatted_min_val_64 = pd.to_datetime(min_val)
            formatted_min_val_str = datetime.datetime.strftime(formatted_min_val_64, '%Y-%m-%d %H:%M:%S')
            formatted_min_val = datetime.datetime.strptime(formatted_min_val_str, '%Y-%m-%d %H:%M:%S').date()
            formatted_max_val_str = datetime.datetime.strftime(max_val, '%Y-%m-%d %H:%M:%S')
            formatted_max_val = datetime.datetime.strptime(formatted_max_val_str, '%Y-%m-%d %H:%M:%S').date()

            min_val = formatted_min_val
            max_val = formatted_max_val

            selected_range = st.sidebar.slider(
                column_friendly_name,
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                format="YYYY-MM-DD",
                key=f"range_filter_{column}"
            )
            
            filters_for_reset[column] = {
                    'key': f"range_filter_{column}",
                    'column': column,
                    'dtype': data[column].dtype,
                    'min': min_val,
                    'max': max_val,
                    'selected_range': selected_range
                }
            
            #filters_for_reset['key'] = ['key': f"range_filter_{column}", 'column': column, 'dtype': data[column].dtype, 'min': min_val, 'max': max_val, 'selected_range': selected_range]
            #print(filters_for_reset['key'])
            #filters_for_reset['column'] = column
            #filters_for_reset['type'] = data[column].dtype
            #filters_for_reset['min'] = min_val
            #filters_for_reset['max'] = max_val
            #filters_for_reset['selected_range'] = selected_range
            
            filters[column] = selected_range

        elif data[column].dtype ==bool:
            
           
            selected_value = st.sidebar.checkbox(
                column_friendly_name,
                value=False,
                key=f"checkbox_filter_{column}"
            )
            if selected_value:
                filters[column] = True

            #filters_for_reset['key'] = 'checkbox_filter_' + {column}
            #filters_for_reset['column'] = column
            #filters_for_reset['type'] = data[column].dtype
            #filters_for_reset['min'] = False
            #filters_for_reset['max'] = True
            #filters_for_reset['selected_range'] = selected_value

            filters_for_reset[column] = {
                    'key': f"checkbox_filter_{column}",
                    'column': column,
                    'dtype': data[column].dtype,
                    'min': min_val,
                    'max': max_val,
                    'selected_range': selected_range
                }

        else:
            unique_values = data[column].dropna().unique().tolist()
            unique_values.sort()
            unique_values.insert(0, ' All')  # Add an option to select all values

            for old_column, new_column in column_rename_for_filter.items():
                    if column == old_column:
                        column_friendly_name = new_column

            if column not in (exclusionList):
                selected_value = st.sidebar.selectbox(
                    column_friendly_name,
                    unique_values,
                    key=f"filter_{column}"
            )

                filters_for_reset[column] = {
                    'key': f"filter_{column}",
                    'column': column,
                    'dtype': data[column].dtype,
                    'min': min_val,
                    'max': max_val,
                    'selected_range': selected_value
                }

            # Add the selected value to the filters dictionary if it's not 'All'
                if selected_value != ' All':
                    filters[column] = selected_value

    # Apply the filters dynamically
    filtered_data = data.copy()

    for column, value in filters.items():
        if isinstance(value, tuple):  # Range filter

            if is_datetime64_any_dtype(data[column]):
                filtered_data[column] = filtered_data[column].dt.date

                filtered_data = filtered_data[
                    (((filtered_data[column]) >= value[0]) & (filtered_data[column] <= value[1])) | (filtered_data[column].isna())
                ]
            else:
                filtered_data = filtered_data[
                    ((filtered_data[column] >= value[0]) & (filtered_data[column] <= value[1])) | (filtered_data[column].isna())
                ]
            print(f"Length of filtered data after range filter on {column}: {len(filtered_data)}")
        else:  # Exact match filter
            filtered_data = filtered_data[
                (filtered_data[column] == value) | (filtered_data[column].isna())
            ]
            print(f"Length of filtered data after exact match filter on {column}: {len(filtered_data)}")
    # Display the filtered results

    #for key, filter_details in filters_for_reset.items():
    #    key = filter_details.get('key')
    #    column = filter_details.get('column')
    #    dtype = filter_details.get('dtype')
    #    min_val = filter_details.get('min')
    #    max_val = filter_details.get('max')
    #    selected_range = filter_details.get('selected_range')
        
    #    print(f"Key: {key}, Column: {column}, Type: {dtype}, Min: {min_val}, Max: {max_val}, Selected Range: {selected_range}")
        # Reset the filters in session state
        
    #if st.button('Reset Filters'):
#    # Reset all filters in session state
    #    for key, filter_details in filters_for_reset.items():
    #        print([filter_details['key']])
    #        if filter_details['key'].startswith('range_filter_'):
    #            st.session_state[filter_details['key']] = (filter_details['min'], filter_details['max'])
    #        elif filter_details['key'].startswith('checkbox_filter_'):
    #            st.session_state[filter_details['key']] = False
    #        elif filter_details['key'].startswith('filter_'):
    #            st.session_state[filter_details['key']] = ' All'
        #st.session_state.clear()  # Clear all session state variables
       # resultsDriverName = 'All'  # Reset the filter to 'All'
        # Simulate a rerun by setting query parameters
    #    st.experimental_rerun()
        #st.rerun()

    st.write(f"Number of filtered results: {len(filtered_data):,d}")
    filtered_data = filtered_data.sort_values(by=['grandPrixYear', 'resultsFinalPositionNumber'], ascending=[False, True])
    filtered_data = filtered_data.drop_duplicates()
    st.dataframe(filtered_data, column_config=columns_to_display, column_order=['grandPrixYear', 'grandPrixName', 'streetRace', 'trackRace', 'constructorName', 'resultsDriverName', 'resultsPodium', 'resultsTop5',
         'resultsTop10','resultsStartingGridPositionNumber','resultsFinalPositionNumber','positionsGained', 'DNF', 'resultsQualificationPositionNumber',
           'q1End', 'q2End', 'q3Top10', 'averagePracticePosition',  'lastFPPositionNumber','numberOfStops', 'averageStopTime', 'totalStopTime',
           'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalPolePositions', 'resultsReasonRetired',
           'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 
           ], hide_index=True, width=2400, height=600)

    positionCorrelation = filtered_data[[
    'lastFPPositionNumber', 'resultsFinalPositionNumber', 'resultsStartingGridPositionNumber','grandPrixLaps', 'averagePracticePosition', 'DNF', 'resultsTop10', 'resultsTop5', 'resultsPodium', 'streetRace', 'trackRace',
    'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions', 'turns', 'positionsGained', 'q1End', 'q2End', 'q3Top10',  'driverBestStartingGridPosition', 'yearsActive',
    'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalPolePositions', 'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums']].corr(method='pearson')
    ##st.button("Clear multiselect", on_click=clear_multi)

    ## Rename Correlation Rows
    positionCorrelation.index=['Last FP.', 'Final Pos.' ,'Starting Grid Pos.', 'Laps', 'Avg Practice Pos.', 
     'DNF', 'Top 10', 'Top 5', 'Podium', 'Street', 'Track', 'Constructor Race Starts', 'Constructor Total Race Wins', 'Constructor Pole Pos.',
     'Turns', 'Positions Gained', 'Out at Q1', 'Out at Q2', 'Q3 Top 10', 'Best Starting Grid Pos.', 'Years Active',
     'Best Result', 'Total Championship Wins', 'Total Pole Positions', 'Race Entries', 'Race Starts', 'Race Wins',
    'Race Laps', 'Total Podiums']
#if st.button("Reset Filters"):
#    reset_filters()
#    st.experimental_rerun()

    # Add visualizations for the filtered data
    st.subheader("Active Years v. Final Position")
    st.scatter_chart(filtered_data, x='resultsFinalPositionNumber', x_label='Final Position', y='yearsActive', y_label='Years Active', use_container_width=True)
    
    st.subheader("Positions Gained")
    st.line_chart(filtered_data, x='short_date', x_label='Date', y='positionsGained', y_label='Positions Gained', use_container_width=True)
    st.scatter_chart(filtered_data, x='short_date', x_label='Date', y='positionsGained', y_label='Positions Gained', use_container_width=True)
    
    st.subheader("Practice Position vs Final Position")
   
    st.scatter_chart(filtered_data, x='lastFPPositionNumber', x_label='Last FP Position', y='resultsFinalPositionNumber', y_label='Final Position', use_container_width=True)

    st.subheader("Starting Position vs Final Position")
    st.scatter_chart(filtered_data, x='resultsStartingGridPositionNumber', x_label='Starting Position', y='resultsFinalPositionNumber', y_label='Final Position', use_container_width=True)

    st.subheader("Average Practice Position vs Final Position")
    st.scatter_chart(filtered_data, x='averagePracticePosition', x_label='Average Practice Position', y='resultsFinalPositionNumber', y_label='Final Position', use_container_width=True)

    # Perform linear regression

    ## use average to fill NaN values
    x_avg = filtered_data['averagePracticePosition'].mean()
    y_avg = filtered_data['resultsFinalPositionNumber'].mean()
    x = filtered_data['averagePracticePosition'].fillna(x_avg)
    y = filtered_data['resultsFinalPositionNumber'].fillna(y_avg)

    # Ensure both x and y have the same length
    if len(x) == len(y) and len(x) > 0:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Create a scatter plot with the regression line
        st.subheader("Linear Regression: Average Practice Position vs Final Position")
        fig, ax = plt.subplots()
        ax.scatter(x, y, label="Data Points", color="blue")
        ax.plot(x, slope * x + intercept, color="red", label=f"y = {slope:.2f}x + {intercept:.2f}")
        ax.set_xlabel("Average Practice Position")
        ax.set_ylabel("Final Position")
        ax.legend()
        st.pyplot(fig, use_container_width=False)
        st.write(f"**Regression Equation:** y = {slope:.2f}x + {intercept:.2f}")

        avg_practice_position_vs_final_position_regression = (f"{slope:.2f}x + {intercept:.2f}")
        #st.write(avg_practice_position_vs_final_position_regression)
        avg_practice_position_vs_final_position_slope = slope
        avg_practice_position_vs_final_position_intercept = intercept
        # Display regression statistics
        st.write(f"**Regression Statistics:**")
        st.write(f"R-squared: {r_value**2:.2f}")
        #st.write(f"P-value: {p_value:.2e}")
        #st.write(f"Standard Error: {std_err:.2f}")
    else:
        st.write("Not enough data for regression analysis.")


    ## use average to fill in NaN values
    x_avg = filtered_data['resultsStartingGridPositionNumber'].mean()
    y_avg = filtered_data['resultsFinalPositionNumber'].mean()
    x = filtered_data['resultsStartingGridPositionNumber'].fillna(x_avg)
    y = filtered_data['resultsFinalPositionNumber'].fillna(y_avg)

    # Ensure both x and y have the same length
    if len(x) == len(y) and len(x) > 0:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Create a scatter plot with the regression line
        st.subheader("Linear Regression: Starting Position vs. Final Position")
        fig, ax = plt.subplots()
        ax.scatter(x, y, label="Data Points", color="blue")
        ax.plot(x, slope * x + intercept, color="red", label=f"y = {slope:.2f}x + {intercept:.2f}")
        ax.set_xlabel("Starting Position")
        ax.set_ylabel("Final Position")
        ax.legend()
        st.pyplot(fig, use_container_width=False)
        st.write(f"**Regression Equation:** y = {slope:.2f}x + {intercept:.2f}")
        starting_vs_final_position_slope = slope
        starting_vs_final_position_intercept = intercept
        starting_vs_final_position_regression = (f"{slope:.2f}x + {intercept:.2f}")

        # Display regression statistics
        st.write(f"**Regression Statistics:**")
        st.write(f"R-squared: {r_value**2:.2f}")
        #st.write(f"P-value: {p_value:.2e}")
        #st.write(f"Standard Error: {std_err:.2f}")
    else:
        st.write("Not enough data for regression analysis.")

    correlation_matrix = positionCorrelation.style.map(highlight_correlation, subset=positionCorrelation.columns[1:])
    
    st.subheader("Correlation Matrix")
    st.caption("Correlation values range from -1 to 1, where -1 indicates a perfect negative correlation, 0 indicates no correlation, and 1 indicates a perfect positive correlation.")
    # Display the correlation matrix
    st.dataframe(correlation_matrix, column_config=correlation_columns_to_display, width=800, height=600)

    driver_performance = filtered_data.groupby(['grandPrixYear', 'resultsDriverName']).agg(
        average_final_position=('resultsFinalPositionNumber', 'mean'),
        total_podiums=('resultsPodium', 'sum')
    ).reset_index()
    
    st.subheader("Driver Performance Over Time")

    # Create an Altair chart for driver performance
    chart = alt.Chart(driver_performance).mark_line().encode(
        x=alt.X('grandPrixYear:O', title='Year', axis=alt.Axis(format='d')),  # Format x-axis as integers
        y=alt.Y('average_final_position', title='Average Final Position'),

        color='resultsDriverName',  # Different lines for each driver
        tooltip=['grandPrixYear', 'resultsDriverName', 'average_final_position']  # Add tooltips for interactivity
    ).properties(width=800, height=400)

    # Display the Altair chart
    st.altair_chart(chart, use_container_width=True)

    constructor_performance = filtered_data.groupby(['grandPrixYear', 'constructorName']).agg(
    total_wins=('resultsFinalPositionNumber', lambda x: (x == 1).sum()),
    total_podiums=('resultsPodium', 'sum'),
    total_pole_positions=('constructorTotalPolePositions', 'sum')
    ).reset_index()
    
    

    
    st.subheader("Constructor Dominance Over the Years")
    st.bar_chart(constructor_performance, x='grandPrixYear', y=['total_wins', 'total_podiums'], color='constructorName', x_label='Year', y_label='Wins and Podiums', use_container_width=True)

    st.subheader("Impact of Starting Grid Position on Final Position")
    st.scatter_chart(filtered_data, x='resultsStartingGridPositionNumber', x_label='Starting Pos.', y_label='Final Pos.', y='resultsFinalPositionNumber', use_container_width=True)

    st.subheader("Pit Stop Analysis")
    st.scatter_chart(filtered_data, x='averageStopTime', x_label='Avg. Stop Time', y='resultsFinalPositionNumber', y_label='Final Pos.', use_container_width=True)

    driver_vs_constructor = filtered_data.groupby(['constructorName', 'resultsDriverName']).agg(
    positionsGained=('positionsGained', 'sum'),
    average_final_position=('resultsFinalPositionNumber', 'mean')
    ).reset_index()
    
    st.subheader("Driver vs Constructor Performance")
    driver_vs_constructor['average_final_position'] = driver_vs_constructor['average_final_position'].round(2)

    driver_vs_constructor = driver_vs_constructor.sort_values(by='average_final_position', ascending=True)
    st.dataframe(driver_vs_constructor, hide_index=True, column_config=driver_vs_constructor_columns_to_display, width=800,
    height=600,)

    dnf_reasons = filtered_data[filtered_data['DNF']].groupby('resultsReasonRetired').size().reset_index(name='count')
    st.subheader("Reasons for DNFs")
    st.bar_chart(dnf_reasons, x='resultsReasonRetired', x_label='Reason', y='count', y_label='Count', use_container_width=True)

    # Crate a table for percentage of DNFs per race entries
    #dnf_percentage = #filtered_data.groupby('resultsDriverName').agg(
        #total_entries=('driverTotalRaceEntries', 'first'),
    dnf_counts = (
    filtered_data[filtered_data['DNF']]
    .groupby(['resultsDriverName', 'driverTotalRaceEntries'])
    .size()
    .reset_index(name='dnf_count')
)
# Calculate DNF percentage
    dnf_counts['dnf_pct'] = (dnf_counts['dnf_count'] / dnf_counts['driverTotalRaceEntries'] * 100).round(1)

    dnf_counts = dnf_counts.sort_values(by='dnf_pct', ascending=False)
    st.subheader("DNF by Driver")

    st.dataframe(
    dnf_counts,
    column_order=['resultsDriverName', 'driverTotalRaceEntries', 'dnf_count', 'dnf_pct'],
    hide_index=True,
    width=800,
    height=600, 
    column_config={
        'resultsDriverName': st.column_config.TextColumn("Driver"),
        'driverTotalRaceEntries': st.column_config.NumberColumn("Total Race Entries", format="%d"),
        'dnf_count': st.column_config.NumberColumn("DNF Count", format="%d"),
        'dnf_pct': st.column_config.NumberColumn("DNF Percentage (%)", format="%.1f")
        },   
    )    

    # Count the number of entries (drivers) for each race
    race_entry_counts = (
        filtered_data
        #.groupby(['grandPrixName', 'grandPrixYear'])
        .groupby(['grandPrixName'])
        .size()
        .reset_index(name='race_entry_count')
)

    #st.subheader("Number of Entries per Race")
    #st.dataframe(race_entry_counts, hide_index=True)

    race_dnf_counts = (
    filtered_data[filtered_data['DNF']]
    #.groupby(['grandPrixName', 'grandPrixYear'])
    .groupby(['grandPrixName'])
    .size()
    .reset_index(name='dnf_count')
)

# Merge with total entries
    race_dnf_stats = pd.merge(race_entry_counts, race_dnf_counts, on=['grandPrixName'], how='left')
    race_dnf_stats['dnf_count'] = race_dnf_stats['dnf_count'].fillna(0).astype(int)
    race_dnf_stats['dnf_pct'] = (race_dnf_stats['dnf_count'] / race_dnf_stats['race_entry_count'] * 100).round(1)

    st.subheader("DNF by Race")
    race_dnf_stats = race_dnf_stats.sort_values(by='dnf_pct', ascending=False)

    st.dataframe(race_dnf_stats, hide_index=True, width=800,
    height=600, column_order=['grandPrixName', 'race_entry_count', 'dnf_count', 'dnf_pct'],
    column_config={
        'grandPrixName': st.column_config.TextColumn("Grand Prix"),
        'race_entry_count': st.column_config.NumberColumn("Total # of Entrants", format="%d"),
        'dnf_count': st.column_config.NumberColumn("DNF Count", format="%d"),
        'dnf_pct': st.column_config.NumberColumn("DNF Percentage (%)", format="%.1f")
        }            
        )
    
    # Count the number of entries (constructors) for each race
    constructor_entry_counts = (
        filtered_data.groupby(['constructorName'])
        .size()
        .reset_index(name='constructor_entry_count')
)

    #st.subheader("Number of Entries per Race")
    #st.dataframe(race_entry_counts, hide_index=True)

    constructor_dnf_counts = (
    filtered_data[filtered_data['DNF']]
    #.groupby(['grandPrixName', 'grandPrixYear'])
    .groupby(['constructorName'])
    .size()
    .reset_index(name='dnf_count')
)

# Merge with total entries
    constructor_dnf_stats = pd.merge(constructor_entry_counts, constructor_dnf_counts, on=['constructorName'], how='left')

    constructor_dnf_stats['dnf_count'] = constructor_dnf_stats['dnf_count'].fillna(0).astype(int)
    constructor_dnf_stats['dnf_pct'] = (constructor_dnf_stats['dnf_count'] / constructor_dnf_stats['constructor_entry_count'] * 100).round(1)

    st.subheader("DNF by Constructor")
    constructor_dnf_stats = constructor_dnf_stats.sort_values(by='dnf_pct', ascending=False)

    st.dataframe(constructor_dnf_stats, hide_index=True, width=800,
    height=600, column_order=['constructorName', 'constructor_entry_count', 'dnf_count', 'dnf_pct'],
    column_config={
        'constructorName': st.column_config.TextColumn("Constructor"),
        'constructor_entry_count': st.column_config.NumberColumn("# of Drivers Entered", format="%d"),
        'dnf_count': st.column_config.NumberColumn("DNF Count", format="%d"),
        'dnf_pct': st.column_config.NumberColumn("DNF Percentage (%)", format="%.1f")
        }            
        )

    st.subheader("DNF Reasons")
    # Create a pie chart for DNF reasons
    dnf_pie_chart = alt.Chart(dnf_reasons).mark_arc().encode(
    theta=alt.Theta(field='count', type='quantitative', title='Count'),
    color=alt.Color(field='resultsReasonRetired', type='nominal', title='Reason'),
    tooltip=['resultsReasonRetired', 'count']  # Add tooltips for interactivity
    ).properties(width=400,height=400)

    # Display the pie chart
    st.altair_chart(dnf_pie_chart, use_container_width=True)

    st.subheader("Track Characteristics and Performance")
    st.scatter_chart(filtered_data, x='turns', y='resultsFinalPositionNumber', use_container_width=True, x_label='Turns', y_label='Final Position')

    season_summary = filtered_data[filtered_data['grandPrixYear'] == current_year].groupby('resultsDriverName').agg(
    positions_gained =('positionsGained', 'sum'),
    total_podiums=('resultsPodium', 'sum')
    ).reset_index()
    
    st.subheader(f"{current_year} Season Summary")
    st.dataframe(season_summary, hide_index=True, column_config=season_summary_columns_to_display, width=800,
    height=600,)

    driver_consistency = filtered_data.groupby('resultsDriverName').agg(
    finishing_position_std=('resultsFinalPositionNumber', 'std')
    ).reset_index()
    
    st.subheader("Driver Consistency")
    st.caption("(Lower is Better)")
    st.bar_chart(driver_consistency, x='resultsDriverName', x_label='Driver', y_label='Standard Deviation - Finishing', y='finishing_position_std', use_container_width=True, )
    driver_consistency = driver_consistency.sort_values(by='finishing_position_std', ascending=True)
    st.caption("Lower standard deviation indicates more consistent finishing positions.")
    st.dataframe(driver_consistency, hide_index=True, column_config={'resultsDriverName': st.column_config.TextColumn("Driver"),
        'finishing_position_std': st.column_config.NumberColumn("Standard Deviation", format="%.3f"),}, width=800, height=600,)

    st.subheader("Predictive Data Model")
    #st.write(f"Total number of results: {len(data):,d}")
    # model, mse, r2, mae = train_and_evaluate_model(filtered_data)

    # st.write(f"Mean Squared Error: {mse:.3f}")

    # st.write(f"R^2 Score: {r2:.3f}")
    # st.write(f"Mean Absolute Error: {mae:.2f}")

    # Extract features and target
    X, y = get_features_and_target(filtered_data)

    if len(X) == 0 or len(y) == 0:
        st.warning("No data available after filtering. Please adjust your filters.")
    else:
        # Split the data
        model, mse, r2, mae = train_and_evaluate_model(filtered_data)

        st.write(f"Mean Squared Error: {mse:.3f}")

        st.write(f"R^2 Score: {r2:.3f}")
        st.write(f"Mean Absolute Error: {mae:.2f}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Create a DataFrame to display the features and predictions
        results_df = X_test.copy()
        results_df['Actual'] = y_test.values
        results_df['Predicted'] = y_pred
        results_df['Error'] = results_df['Actual'] - results_df['Predicted']

        # st.write(f"results_df shape: {results_df.shape}")
        # st.write(f"X_test shape: {X_test.shape}")
        # st.write(f"Filtered data shape: {filtered_data.shape}")

        # Display the first 30 rows
        st.subheader("First 30 Results with Accuracy")
        st.dataframe(results_df[['grandPrixName', 'constructorName', 'resultsDriverName','Actual', 'Predicted', 'Error']].head(30), hide_index=True, #width=800, height=600,
                     column_order=['grandPrixName', 'constructorName', 'resultsDriverName', 'Actual', 'Predicted', 'Error'],
                     column_config={
                        'grandPrixName': st.column_config.TextColumn("Grand Prix"),
                        'constructorName': st.column_config.TextColumn("Constructor"),
                        'resultsDriverName': st.column_config.TextColumn("Driver"),
                        'Actual': st.column_config.NumberColumn("Actual Pos.", format="%d"),
                        'Predicted': st.column_config.NumberColumn("Predicted Pos.", format="%.3f"),
                        'Error': st.column_config.NumberColumn("Error", format="%.3f")})
        # st.dataframe(results_df.head(15), hide_index=True, width=800, height=600,)
        # Display feature importances
        st.subheader("Feature Importance")
        
        # Retrieve feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()

        # Retrieve feature importances
        feature_importances = model.named_steps['regressor'].feature_importances_


        # Clean up feature names by removing 'num__'
        feature_names = [name.replace('num__', '') for name in feature_names]
        feature_names = [name.replace('cat__', '') for name in feature_names]

        # Create a DataFrame for feature importances
        feature_importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances,
            'Percentage': feature_importances / feature_importances.sum() * 100,
            #'Cumulative Percentage': np.cumsum(feature_importances / feature_importances.sum() * 100),
            #'Feature Type': ['Categorical' if 'cat' in name else 'Numerical' for name in feature_names]
        }).sort_values(by='Importance', ascending=False)

        # Display the top 50 features
        st.dataframe(feature_importances_df.head(50), hide_index=True, width=800)
        # Display the predictive data model without index
        # -----------------------------

if st.checkbox(f"Show {current_year} Schedule"):
    st.title(f"{current_year} Races:")
    
    raceSchedule = raceSchedule[raceSchedule['year'] == current_year]
    #st.dataframe(raceSchedule)
    #st.write(raceSchedule.columns)
    #st.dataframe(race_messages)
    
    st.write(f"Total number of races: {len(raceSchedule)}")
    
    st.dataframe(raceSchedule, column_config=schedule_columns_to_display,
    #st.dataframe(race_messages, column_config=schedule_columns_to_display,
        hide_index=True,  width=800, height=600, column_order=['round', 'fullName', 'date', 'time', 
        'circuitType', 'courseLength', 'laps', 'turns', 'distance', 'totalRacesHeld'])


if st.checkbox("Show Next Race"):
#### fix to show current day's race
    st.subheader("Next Race:")
    
    # include the current date in the raceSchedule
    raceSchedule['date_only'] = pd.to_datetime(raceSchedule['date']).dt.date
    nextRace = raceSchedule[raceSchedule['date_only'] >= datetime.datetime.today().date()]

    # Create a copy of the slice to avoid the warning 
    nextRace = nextRace.sort_values(by=['date'], ascending = True).head(1).copy()
    
    st.dataframe(nextRace, width=800, column_config=next_race_columns_to_display, hide_index=True, 
        column_order=['date', 'time', 'fullName', 'courseLength', 'turns', 'laps'])
    
    # Limit detailsOfNextRace by the grandPrixId of the next race
    next_race_id = nextRace['grandPrixId'].head(1).values[0]
    upcoming_race = pd.merge(nextRace, raceSchedule, left_on='grandPrixId', right_on='grandPrixId', how='inner', suffixes=('_nextrace', '_races'))
    #st.write(upcoming_race_id.columns)
    upcoming_race = upcoming_race.sort_values(by='date_nextrace', ascending = False).head(1).copy()
    upcoming_race_id = upcoming_race['id_grandPrix_nextrace'].unique().copy()

    st.subheader("Past Results:")
    detailsOfNextRace = data[data['grandPrixRaceId'] == next_race_id]

    # Sort detailsOfNextRace by grandPrixYear descending and resultsFinalPositionNumber ascending
    detailsOfNextRace = detailsOfNextRace.sort_values(by=['grandPrixYear', 'resultsFinalPositionNumber'], ascending=[False, True])

    st.write(f"Total number of results: {len(detailsOfNextRace)}")
    #detailsOfNextRace = detailsOfNextRace.drop_duplicates()
    detailsOfNextRace = detailsOfNextRace.drop_duplicates(subset=['resultsDriverName', 'grandPrixYear'])
    st.dataframe(detailsOfNextRace, column_config=columns_to_display, hide_index=True)

    #dups = detailsOfNextRace[detailsOfNextRace.duplicated(subset=['resultsDriverName', 'grandPrixYear'], keep=False)]
    #st.caption("Dups")
    #st.write(dups)
    
    last_race = detailsOfNextRace.iloc[1]
    #st.write(last_race.columns)
    
    #st.write("current race ID")
    #st.write(upcoming_race_id)
    #st.write("Last race ID")
    #st.dataframe(last_race)

    active_drivers = data[data['activeDriver'] == True]
    active_drivers = active_drivers['resultsDriverId'].unique()
    # st.write(f"Total number of active drivers: {len(active_drivers)}")
    # st.write(active_drivers.tolist())
    input_data = detailsOfNextRace.copy()
    
    input_data = input_data[input_data['raceId_results'] == last_race['raceId_results']]

    features, _ = get_features_and_target(data)
    feature_names = features.columns.tolist()

    
    #st.write(last_race['raceId'])
    #st.write(next_race_id)
    #st.write(upcoming_race_id)
   # st.write(type(upcoming_race_id))
    if next_race_id not in practices['raceId'].values:
        # The upcoming race is NOT in the practices dataset
        practices = practices[practices['raceId'] == last_race['raceId_results']]
        qualifying = qualifying[qualifying['raceId'] == last_race['raceId_results']]
    else:    
        # The upcoming race IS in the practices dataset
        practices = practices[practices['raceId'] == upcoming_race_id[0]]
        qualifying = qualifying[qualifying['raceId'] == upcoming_race_id[0]]

    if nextRace['freePractice2Date'].isnull().all():
        practices = practices[practices['Session'] =='FP1']
    else:
        practices = practices[practices['Session'] =='FP2']    


    driver_inputs = []

    ###### This only includes drivers who had results in the most recent race

    # When building driver_inputs, select features + resultsDriverId for reference
    for driver in active_drivers:
        driver_data = input_data[input_data['resultsDriverId'] == driver]
        if len(driver_data) > 0:
            # Keep resultsDriverId for reference, but do not use it for prediction
            driver_inputs.append(driver_data[feature_names + ['resultsDriverId']])
        #     st.write(f"Driver {driver} data added with {len(driver_data)} rows.")
        # else:
        #     st.write(f"Driver {driver} has no data for the race.")    

    all_active_driver_inputs = pd.concat(driver_inputs, ignore_index=True)

    all_active_driver_inputs = pd.merge(all_active_driver_inputs, practices, left_on='resultsDriverId', right_on='driverId', how='left')
    #st.write(all_active_driver_inputs.columns)
    # st.write("Active driver inputs")
    # st.dataframe(all_active_driver_inputs)

    # List of columns you want to clean up
    columns_to_clean = [
        'LapTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec',
        'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph',
        'best_theory_lap_sec', 'Session'
    ]

    # Build a rename dict for _x and _y suffixes
    rename_dict = {}
    for col in columns_to_clean:
        for suffix in ['_x', '_y']:
            if f"{col}{suffix}" in all_active_driver_inputs.columns:
                rename_dict[f"{col}{suffix}"] = col

    all_active_driver_inputs = all_active_driver_inputs.rename(columns=rename_dict)            
    

    # all_active_driver_inputs = pd.merge(
    # all_active_driver_inputs,
    # drivers[['id', 'abbreviation']],
    # left_on='resultsDriverId',
    # right_on='id',
    # how='left'
    # )
    # st.dataframe(all_active_driver_inputs)

    all_active_driver_inputs = pd.merge(
        all_active_driver_inputs, 
        qualifying, 
        left_on='abbreviation', 
        right_on='Abbreviation', 
        how='inner',
        suffixes=('_datamodel', '_qualifying')
    )
    
    # st.dataframe(all_active_driver_inputs)
    all_active_driver_inputs = all_active_driver_inputs.rename(columns={'Points_datamodel': 'Points', 'totalChampionshipPoints_datamodel': 'totalChampionshipPoints',
        'totalPolePositions_datamodel': 'totalPolePositions','totalFastestLaps_datamodel': 'totalFastestLaps'} )    

    # st.write(qualifying.columns)
    # st.dataframe(qualifying)
    # st.write("With qualifying data")
    # st.dataframe(all_active_driver_inputs)
    # st.write(all_active_driver_inputs.columns)
    # all_active_driver_inputs = pd.merge(all_active_driver_inputs, qualifying, left_on='resultsDriverId', right_on='driverId', how='left')

    # st.dataframe(all_active_driver_inputs)

    #st.dataframe(all_active_driver_inputs, hide_index=True, width=800, height=600)
    ## Pull the most recent data for practices and qualifying

    #st.write(all_active_driver_inputs.columns)
    #all_active_driver_inputs['bestQualifyingTime_sec'] =  all_active_driver_inputs['bestQualifyingTime_sec'].fillna(all_active_driver_inputs['bestQualifyingTime_sec'].mean())

    # get the most recent qualifying data for this race
    
    #qualifying = qualifying[qualifying['grandPrixId'] == next_race_id ]
   #st.write(qualifying['date'])
    # active_drivers

    

    ## use last year's data in the absence of current data
    ## include the best sector and best theory sector as well as current practice and qualifying rank
    ## set the null values to the previous year's data in the first instance, and to the average if no previous data exists
    ## modify the dataframe for the learning model to point to new data if available


    ## redo this to remove all of the ergast pulls, use the data from the csv files


    # if pd.to_datetime(nextRace['date'].iloc[0]).date() >= datetime.datetime.today().date():
    #     # means the qualifying data has already past and can be run through Ergast
    #     st.write("Qualifying data for", nextRace['id_grandPrix'].iloc[0], "and", nextRace['year'].iloc[0])
    #     # now pull the qualifying data through ergast
    #     ## write a check to see if the .csv exists. If not, pull the data from Ergast
    #     if not path.exists(path.join(DATA_DIR, f"qualifying_{nextRace['id_grandPrix'].iloc[0]}_{nextRace['year'].iloc[0]}.csv")):
    #         st.write("Qualifying data for", nextRace['id_grandPrix'].iloc[0], "and", nextRace['year'].iloc[0], "not found. Pulling from Ergast.")

    #         # First, try FastF1 to get the data. If it's throwing an error, use last year's data

    #         try:
    #             qualifying_results_list = []
    #             qualifying = fastf1.get_session(nextRace['year'].iloc[0], nextRace['round'].iloc[0], 'Q')
    #             qualifying.load()

    #                 # Get qualifying results as a DataFrame
    #             qualifying_results = qualifying.results

    #         except Exception as e:
    #             st.error(f"Error fetching qualifying data: {e}")
    #             qualifying_results = qualifying[qualifying['raceId'] == last_race['raceId_results']]
    #             #last_race['raceId_results']
    #             #qualifying_results_df = pd.DataFrame(columns=['Driver', 'Q1', 'Q2', 'Q3', 'Q1_sec', 'Q2_sec', 'Q3_sec', 'Round', 'Year', 'Event'])

    #         if isinstance(qualifying_results, pd.DataFrame):
    #                 # Add metadata to the DataFrame
    #             qualifying_results['Round'] = nextRace['round'].iloc[0]
    #             qualifying_results['Year'] = nextRace['year'].iloc[0]
    #             qualifying_results['Event'] = qualifying.event['EventName']
    #             qualifying_results_list.append(qualifying_results)
    #     #nextRace = nextRace.sort_values(by=['date'], ascending = True).head(1).copy()
    #             qualifying_results_df = pd.concat(qualifying_results_list, ignore_index=True)
    #             st.dataframe(qualifying_results_df)

    #             for col in ['Q1', 'Q2', 'Q3']:
    #                 if col in qualifying_results_df.columns:
    #                     qualifying_results_df[f'{col}_sec'] = pd.to_timedelta(qualifying_results_df[col]).dt.total_seconds()

    #             year = str(nextRace['year'].iloc[0])
    #             filename = f"qualifying_{nextRace['id_grandPrix'].iloc[0]}_{year}.csv"
    #             qualifying_results_df.to_csv(path.join(DATA_DIR, filename), sep='\t', index=False)

            

    #     all_practice_laps = []

    #     if not path.exists(path.join(DATA_DIR, f"practices_{nextRace['id_grandPrix'].iloc[0]}_{nextRace['year'].iloc[0]}.csv")):
    #         for session_type in ['FP1', 'FP2', 'FP3']:
    #             #session_key = (year, round_number, session_type)
    #             #session_date = pd.to_datetime(season_schedule.iloc[round_number - 1]['raceDate'])
                
    #             session = fastf1.get_session(nextRace['year'].iloc[0], nextRace['round'].iloc[0], session_type)
    #             session.load()
    #             session_drivers = session.drivers

    #             for driver in session_drivers:
    #                 laps = session.laps.pick_drivers(driver)
    #                 fastest_lap = laps.pick_fastest()
    #                 if fastest_lap is not None and not fastest_lap.empty:
    #                     fastest_lap = fastest_lap.copy()
    #                     fastest_lap['Year'] = session.date.year  
    #                     fastest_lap['FP_Name'] = session.event['EventName']
    #                     fastest_lap['Round'] = session.event['RoundNumber']
    #                     fastest_lap['Session'] = session_type

    #                     # Safely get best sector times
    #                     if 'Sector1Time' in laps.columns and not laps['Sector1Time'].isnull().all():
    #                         best_s1 = laps.loc[laps['Sector1Time'].idxmin()]
    #                         fastest_lap['best_s1'] = best_s1['Sector1Time']
    #                     else:
    #                         fastest_lap['best_s1'] = pd.NaT
    #                     if 'Sector2Time' in laps.columns and not laps['Sector2Time'].isnull().all():
    #                         best_s2 = laps.loc[laps['Sector2Time'].idxmin()]
    #                         fastest_lap['best_s2'] = best_s2['Sector2Time']
    #                     else:
    #                         fastest_lap['best_s2'] = pd.NaT
    #                     if 'Sector3Time' in laps.columns and not laps['Sector3Time'].isnull().all():
    #                         best_s3 = laps.loc[laps['Sector3Time'].idxmin()]
    #                         fastest_lap['best_s3'] = best_s3['Sector3Time']
    #                     else:
    #                         fastest_lap['best_s3'] = pd.NaT

    #                     # Only calculate theoretical lap if all sectors are present
    #                     if pd.notnull(fastest_lap['best_s1']) and pd.notnull(fastest_lap['best_s2']) and pd.notnull(fastest_lap['best_s3']):
    #                         fastest_lap['best_theory_lap'] = fastest_lap['best_s1'] + fastest_lap['best_s2'] + fastest_lap['best_s3']
    #                         fastest_lap['best_theory_lap_diff'] = fastest_lap['LapTime'] - (fastest_lap['best_s1'] + fastest_lap['best_s2'] + fastest_lap['best_s3'])
    #                     else:
    #                         fastest_lap['best_theory_lap'] = pd.NaT
    #                         fastest_lap['best_theory_lap_diff'] = pd.NaT

    #                     all_practice_laps.append(fastest_lap)

    #         # Convert all_laps to DataFrame after all loops
    #         if all_practice_laps:
    #             all_practice_laps_df = pd.DataFrame(all_practice_laps)
    #         else:
    #     # Create an empty DataFrame with the expected columns
    #             all_practice_laps_df = pd.DataFrame(columns=[
    #             'Time', 'Driver', 'DriverNumber', 'LapTime', 'LapNumber', 'Stint', 'PitOutTime', 'PitInTime',
    #             'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime',
    #             'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest', 'Compound',
    #             'TyreLife', 'FreshTyre', 'Team', 'LapStartTime', 'LapStartDate', 'TrackStatus', 'Position',
    #             'Deleted', 'DeletedReason', 'FastF1Generated', 'IsAccurate', 'Year', 'FP_Name', 'Round', 'Session',
    #             'best_s1', 'best_s2', 'best_s3', 'best_theory_lap', 'best_theory_lap_diff', 'SpeedI1_mph',
    #             'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'LapTime_sec', 'Sector1Time_sec', 'Sector2Time_sec',
    #             'Sector3Time_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec', 'best_theory_lap_diff_sec'])

    #         # Modify speed to MPH from KM/h
    #         for col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
    #             if col in all_practice_laps_df.columns:
    #                 all_practice_laps_df[f'{col}_mph'] = all_practice_laps_df[col].apply(km_to_miles)

    #         # Convert time columns to seconds from timedelta
    #         for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'best_s1', 'best_s2', 'best_s3', 'best_theory_lap', 'best_theory_lap_diff']:
    #             if col in all_practice_laps_df.columns:
    #                 all_practice_laps_df[f'{col}_sec'] = pd.to_timedelta(all_practice_laps_df[col]).dt.total_seconds()

    #         # Merge with driver info
    #         active_drivers = pd.merge(data, drivers, left_on='resultsDriverId', right_on='id', how='inner')
    #         active_drivers = active_drivers[active_drivers['activeDriver'] == True]
    #         active_drivers = active_drivers[['resultsDriverId', 'driverName', 'abbreviation']].drop_duplicates()


    #         # Merge practice laps with driver names
    #         all_practice_laps_with_driver_names = pd.merge(
    #             active_drivers, 
    #             all_practice_laps_df, 
    #             left_on='abbreviation', 
    #             right_on='Driver', 
    #             how='inner'
    #         )

    #         # Merge with races info
    #         races_with_mapping = pd.merge(
    #             raceSchedule, 
    #             all_practice_laps_with_driver_names, 
    #             left_on=['year', 'round'], 
    #             right_on=['Year', 'Round'], 
    #             how='inner', 
    #             suffixes=('_races', '_mapping')
    #         ).drop_duplicates()

            
    #         year = str(nextRace['year'].iloc[0])
    #         filename = f"practices_{nextRace['id_grandPrix'].iloc[0]}_{year}.csv"
    #         races_with_mapping.to_csv(path.join(DATA_DIR, filename), sep='\t', index=False)




            # Save to CSV
            # races_with_mapping.to_csv(
            #     path.join(DATA_DIR, f'all_practice_laps.csv'),
            #     sep='\t',
            #     index=False
            # )
        

    # For prediction, drop resultsDriverId

    # missing = [col for col in feature_names if col not in all_active_driver_inputs.columns]
    # if missing:
    #     st.error(f"Missing columns in all_active_driver_inputs: {missing}")
    # else:
    #     st.success("All required feature columns are present.")

    # missing = [col for col in feature_names if col not in all_active_driver_inputs.columns]
    # extra = [col for col in all_active_driver_inputs.columns if col not in feature_names]
    # st.write("Missing columns:", missing)
    # st.write("Extra columns:", extra)

    # Remove duplicate columns (if any)
    all_active_driver_inputs = all_active_driver_inputs.loc[:, ~all_active_driver_inputs.columns.duplicated()]
    # st.write(all_active_driver_inputs.columns.tolist())
# # Select columns in the exact order
# X_predict = all_active_driver_inputs[feature_names]

    X_predict = all_active_driver_inputs[feature_names]
    predicted_position = model.predict(X_predict)

    # Get DNF feature names
    dnf_features, _ = get_features_and_target_dnf(data)
    dnf_feature_names = dnf_features.columns.tolist()

    # Build X_predict for DNF using only those columns
    X_predict_dnf = all_active_driver_inputs[dnf_feature_names]

    # Now predict DNF probability
    predicted_dnf_proba = dnf_model.predict_proba(X_predict_dnf)[:, 1]

    # For DNF probability
    predicted_dnf_proba = dnf_model.predict_proba(X_predict)[:, 1]  # Probability of DNF=True

    # Add both to your DataFrame
    all_active_driver_inputs['PredictedFinalPosition'] = predicted_position
    all_active_driver_inputs['PredictedDNFProbability'] = predicted_dnf_proba
    all_active_driver_inputs['PredictedDNFProbabilityPercentage'] = (all_active_driver_inputs['PredictedDNFProbability'] * 100).round(3)
    all_active_driver_inputs['driverDNFCount'] = all_active_driver_inputs['driverDNFCount'].fillna(0).astype(int)
    all_active_driver_inputs['driverDNFAvg'] = all_active_driver_inputs['driverDNFAvg'].fillna(0).astype(float)
    all_active_driver_inputs['driverDNFPercentage'] = (all_active_driver_inputs['driverDNFAvg'].fillna(0).astype(float) * 100).round(3)

    all_active_driver_inputs.sort_values(by='PredictedFinalPosition', ascending=True, inplace=True)
    all_active_driver_inputs['Rank'] = range(1, len(all_active_driver_inputs) + 1)
    all_active_driver_inputs = all_active_driver_inputs.set_index('Rank')

     
    st.subheader("Predictive Results for Active Drivers")

    st.dataframe(all_active_driver_inputs, hide_index=False, column_config=predicted_position_columns_to_display, width=800, height=600, 
    column_order=['constructorName', 'resultsDriverName', 'PredictedFinalPosition'])    

    st.subheader("Predictive DNF")

    all_active_driver_inputs.sort_values(by='PredictedDNFProbability', ascending=False, inplace=True)
    st.dataframe(all_active_driver_inputs, hide_index=False, column_config=predicted_dnf_position_columns_to_display, width=800, height=600, 
    column_order=['constructorName', 'resultsDriverName', 'driverDNFCount',  'driverDNFPercentage', 'PredictedDNFProbabilityPercentage'], )  

   # st.write(predicted_position)
    #st.write(f"Predicted Final Position: {predicted_position[0]:.2f}")
    #st.dataframe(input_data, hide_index=True, width=800, height=600)
    # Fill NaN values with the mean of the respective columns

    #results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['trackRace'] = (
    #results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['circuitType'] == 'RACE')

    # most_recent_row = data.sort_values(by='short_date', ascending=False).iloc[0]
    #most_recent_starting_grid_position = most_recent_row['resultsStartingGridPositionNumber']

    #input_data['resultsStartingGridPositionNumber'] = 

    #input_data = input_data.fillna(input_data.mean())


    individual_race_grouped = detailsOfNextRace.groupby(['resultsDriverName']).agg(
        #activeDriver = ('activeDriver', 'first'),
        average_starting_position=('resultsStartingGridPositionNumber', 'mean'),
    
        average_ending_position=('resultsFinalPositionNumber', 'mean'),
        average_positions_gained=('positionsGained', 'mean'),
        driver_races=('resultsFinalPositionNumber', 'count')
    ).reset_index()

    # Rename the columns for better readability
    individual_race_grouped = individual_race_grouped.sort_values(by=['average_ending_position'], ascending=[True])

    individual_race_grouped_constructor = detailsOfNextRace.groupby(['constructorName']).agg(
        #activeDriver = ('activeDriver', 'first'),
        average_starting_position=('resultsStartingGridPositionNumber', 'mean'),
        average_ending_position=('resultsFinalPositionNumber', 'mean'),
        average_positions_gained=('positionsGained', 'mean'),
        driver_races=('resultsFinalPositionNumber', 'count')
    ).reset_index()

    # Rename the columns for better readability
    individual_race_grouped_constructor = individual_race_grouped_constructor.sort_values(by=['average_ending_position'], ascending=[True])

    

    #dnf_summary_2024 = data[
    #(data['grandPrixYear'] == 2024) &
    #(data['resultsReasonRetired'].notnull()) &
    #(data['resultsReasonRetired'] != '')
    #].groupby('raceId').size().reset_index(name='dnf_count')
    #st.write(dnf_summary_2024)

    st.subheader(f"Flags and Safety Cars from {nextRace['fullName'].head(1).values[0]}:")
    st.caption("Race messages, including flags, are only available going back to 2018.")
    # race_control_messages_grouped_with_dnf.csv
    raceMessagesOfNextRace = race_messages[race_messages['grandPrixId'] == next_race_id]
    #st.write(raceMessagesOfNextRace.columns)
    #raceMessagesOfNextRace = pd.merge(raceMessagesOfNextRace, dnf_summary, on='raceId', how='left')
    raceMessagesOfNextRace = raceMessagesOfNextRace.sort_values(by='Year', ascending = False)

    st.write(f"Total number of results: {len(raceMessagesOfNextRace)}")
    st.dataframe(raceMessagesOfNextRace, hide_index=True, width=800,column_config=flags_safety_cars_columns_to_display, 
                 column_order=['Year', 'Round', 'SafetyCarStatus', 'redFlag', 'yellowFlag', 'doubleYellowFlag', 'dnf_count'])

    st.subheader(f"Driver Performance in {nextRace['fullName'].head(1).values[0]}:")
    st.write(f"Total number of results: {len(individual_race_grouped)}")
    
    # Display the grouped data without index
    st.dataframe(individual_race_grouped, hide_index=True, width=800, height=600, column_config=individual_race_grouped_columns_to_display)

    st.subheader(f"Constructor Performance in {nextRace['fullName'].head(1).values[0]}:")
    st.dataframe(individual_race_grouped_constructor, hide_index=True, width=800, height=600, column_config=individual_race_grouped_columns_to_display)

    weather_with_grandprix = weatherData[weatherData['grandPrixId'] == next_race_id]
    
    st.subheader(f"Weather Data for {weather_with_grandprix['fullName'].head(1).values[0]}:")
    st.write(f"Total number of weather records: {len(weather_with_grandprix)}")

    weather_with_grandprix = weather_with_grandprix.sort_values(by='short_date', ascending = False)
    st.dataframe(weather_with_grandprix, width=800, column_config=weather_columns_to_display, hide_index=True)





if st.checkbox('Show Raw Data'):

    st.write(f"Total number of results: {len(data):,d}")

    st.dataframe(data, column_config=columns_to_display,
        hide_index=True,  width=800, height=600)

if st.checkbox('Show Predictive Data Model'):
    st.subheader("Predictive Data Model")
    model, mse, r2, mae = train_and_evaluate_model(data)

    st.write(f"Mean Squared Error: {mse:.3f}")
    st.write(f"R^2 Score: {r2:.3f}")
    st.write(f"Mean Absolute Error: {mae:.2f}")

    # Extract features and target
    X, y = get_features_and_target(data)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Create a DataFrame to display the features and predictions
    results_df = X_test.copy()
    results_df['Actual'] = y_test.values
    results_df['Predicted'] = y_pred

    # Display the first 15 rows
    st.subheader("Predictive Results with Features")
    st.dataframe(results_df, hide_index=True, width=800)

    # Display feature importances
    st.subheader("Feature Importances")
    
    # Retrieve feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()

    # Clean up feature names by removing 'num__'
    feature_names = [name.replace('num__', '') for name in feature_names]
    feature_names = [name.replace('cat__', '') for name in feature_names]

    # Retrieve feature importances
    feature_importances = model.named_steps['regressor'].feature_importances_

    # Create a DataFrame for feature importances
    feature_importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance (%)': feature_importances * 100
    }).sort_values(by='Importance (%)', ascending=False)

    # Display all features
    st.dataframe(feature_importances_df, hide_index=True, width=800)

if st.checkbox('Show Correlations for all races'):
    st.subheader("Correlation Matrix")
    
    # Rename rows and columns in the correlation matrix
    correlation_matrix = correlation_matrix.rename(
        index={
            'resultsPodium': 'Podium',
            'resultsTop5': 'Top 5',
            'resultsTop10': 'Top 10',
            'resultsStartingGridPositionNumber': 'Starting Grid Position',
            'resultsFinalPositionNumber': 'Final Position',
            'positionsGained': 'Positions Gained',
            'DNF': 'DNF',
            'averagePracticePosition': 'Avg Practice Pos.',
            'grandPrixLaps': 'Laps',
            'lastFPPositionNumber': 'Last FP Pos.',
            'resultsQualificationPositionNumber': 'Qual. Pos.',
            'constructorTotalRaceStarts': 'Constructor Race Starts',
            'constructorTotalRaceWins': 'Constructor Race Wins',
            'constructorTotalPolePositions': 'Constructor Pole Pos.',
            'turns': 'Turns',
            'q1End': 'Out at Q1',
            'q2End': 'Out at Q2',

            'q3Top10': 'Q3 Top 10',
            'numberOfStops': 'Number of Stops'
        }
    )

    # Apply styling to highlight correlations
    correlation_matrix = correlation_matrix.style.map(highlight_correlation, subset=correlation_matrix.columns[1:])
    
    # Display the correlation matrix
    st.dataframe(correlation_matrix, column_config=correlation_columns_to_display, hide_index=True, width=800 , height=600)

    dnf_counts = data[data['DNF']].groupby('resultsDriverName').size().reset_index(name='dnf_count')
    st.dataframe(dnf_counts, hide_index=True)


