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
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
import seaborn as sns
from xgboost import XGBRegressor
import shap
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb

EarlyStopping = xgb.callback.EarlyStopping



DATA_DIR = 'data_files/'

def simulate_rookie_predictions(data, all_active_driver_inputs, current_year, n_simulations=1000):
    """
    Adjust rookie driver predictions using Monte Carlo simulation based on historical rookie results,
    constructor strength, and practice position.
    """
    # Identify rookie drivers (first F1 season or <5 starts)
    # Calculate the number of races in the current season
    current_season_race_count = raceSchedule[raceSchedule['year'] == current_year]['grandPrixId'].nunique()
    # st.write("Current season race count:", current_season_race_count)
    # Rookie mask: drivers with fewer starts than a full season
    rookie_mask = all_active_driver_inputs['driverTotalRaceStarts'] < current_season_race_count

    rookies = all_active_driver_inputs[rookie_mask].copy()

    # Get the current race name
    race_name = rookies['grandPrixName'].iloc[0] if 'grandPrixName' in rookies.columns and len(rookies) > 0 else None

    # Historical rookie results at this track
    historical_rookies = data[
        (data['grandPrixName'] == race_name) &
        (data['yearsActive'] <= 1) &
        (data['grandPrixYear'] < current_year)
    ]

    # If not enough historical rookies, fallback to all tracks
    if len(historical_rookies) < 10:
        historical_rookies = data[
            (data['yearsActive'] <= 1) &
            (data['grandPrixYear'] < current_year)
        ]

    # For each rookie, simulate their predicted position
    for idx, rookie in rookies.iterrows():
        # Sample historical rookie final positions
        hist_positions = historical_rookies['resultsFinalPositionNumber'].dropna()
        if len(hist_positions) < 3:
            # Fallback to all drivers if not enough rookie data
            hist_positions = data['resultsFinalPositionNumber'].dropna()
        mu, sigma = hist_positions.mean(), hist_positions.std()
        # Truncate between 1 and 20 (F1 grid)
        a, b = (1 - mu) / sigma, (20 - mu) / sigma
        sampled_positions = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_simulations)

        # Adjust by constructor strength (lower rank = better team)
        constructor_rank = rookie.get('constructorRank', 10)
        constructor_adj = np.clip(1 + (constructor_rank - 10) * 0.2, 0.7, 1.3)

        # Adjust by practice position (if available)
        practice_adj = 1.0
        if not pd.isna(rookie.get('averagePracticePosition', np.nan)):
            practice_adj = np.clip(rookie['averagePracticePosition'] / 10, 0.7, 1.3)

        # Simulate predicted position
        simulated_positions = sampled_positions * constructor_adj * practice_adj
        predicted = np.median(simulated_positions)

        # Assign to output DataFrame
        # all_active_driver_inputs.at[idx, 'PredictedFinalPosition'] = predicted
        # ...inside the for idx, rookie in rookies.iterrows(): loop...
        col = 'PredictedFinalPosition'
        if col in all_active_driver_inputs.columns:
            dtype = all_active_driver_inputs[col].dtype
            all_active_driver_inputs.at[idx, col] = dtype.type(predicted)
        else:
            all_active_driver_inputs.at[idx, col] = float(predicted)

        col = 'PredictedFinalPositionStd'
        std_value = float(np.std(simulated_positions))
        if col in all_active_driver_inputs.columns:
            dtype = all_active_driver_inputs[col].dtype
            all_active_driver_inputs.at[idx, col] = dtype.type(std_value)
        else:
            all_active_driver_inputs.at[idx, col] = float(std_value)

        # all_active_driver_inputs.at[idx, 'PredictedFinalPositionStd'] = np.std(simulated_positions)

    return all_active_driver_inputs

def simulate_rookie_dnf(data, all_active_driver_inputs, current_year, n_simulations=1000):
    """
    Adjust rookie DNF probability using Monte Carlo simulation based on historical rookie DNFs.
    """
    # Calculate the number of races scheduled in the current season
    current_season_race_count = raceSchedule[raceSchedule['year'] == current_year]['grandPrixId'].nunique()
    # Identify rookies: fewer starts than a full season
    rookie_mask = all_active_driver_inputs['driverTotalRaceStarts'] < current_season_race_count
    rookies = all_active_driver_inputs[rookie_mask].copy()

    # Get current race name
    race_name = rookies['grandPrixName'].iloc[0] if 'grandPrixName' in rookies.columns and len(rookies) > 0 else None

    # Historical rookie DNFs at this track
    historical_rookies = data[
        (data['grandPrixName'] == race_name) &
        (data['yearsActive'] <= 1) &
        (data['grandPrixYear'] < current_year)
    ]

    # If not enough historical rookies, fallback to all tracks
    if len(historical_rookies) < 10:
        historical_rookies = data[
            (data['yearsActive'] <= 1) &
            (data['grandPrixYear'] < current_year)
        ]

    # For each rookie, simulate DNF probability
    for idx, rookie in rookies.iterrows():
        # Sample historical rookie DNFs (1 if DNF, 0 if not)
        hist_dnfs = historical_rookies['DNF'].dropna().astype(int)
        if len(hist_dnfs) < 3:
            # Fallback to all drivers if not enough rookie data
            hist_dnfs = data['DNF'].dropna().astype(int)
        # Monte Carlo simulation
        sampled_dnfs = np.random.choice(hist_dnfs, size=n_simulations, replace=True)
        # Adjust by constructor reliability (lower rank = better team)
        constructor_rank = rookie.get('constructorRank', 10)
        constructor_adj = np.clip(1 - (constructor_rank - 10) * 0.03, 0.85, 1.05)
        # Adjust by practice reliability (if available)
        practice_adj = 1.0
        if not pd.isna(rookie.get('averagePracticePosition', np.nan)):
            practice_adj = np.clip(1 - (rookie['averagePracticePosition'] / 100), 0.85, 1.05)
        # Simulate DNF probability
        simulated_dnf_proba = sampled_dnfs * constructor_adj * practice_adj
        predicted_dnf = np.mean(simulated_dnf_proba)
        # Assign to output DataFrame
        all_active_driver_inputs.at[idx, 'PredictedDNFProbability'] = predicted_dnf
        all_active_driver_inputs.at[idx, 'PredictedDNFProbabilityStd'] = np.std(simulated_dnf_proba)

    return all_active_driver_inputs

# Done to avoid getting an error on Github after upload

if os.environ.get('LOCAL_RUN') == '1':

    fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

st.set_page_config(
   page_title="Formula 1 Analysis",
   page_icon=path.join(DATA_DIR, 'favicon.png'),
   layout="wide",
   initial_sidebar_state="expanded"
)

def km_to_miles(km):
    return km * 0.621371

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
    'delta_from_race_avg': 'Delta from Race Avg. (s)',
    'driverAge': 'Driver Age',
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
    'yearsActive': None, 'bestQualifyingTime_sec': None, 'resultsDriverId': None,
    'PredictedFinalPositionStd': st.column_config.NumberColumn("Rookie Uncertainty (Std)", format="%.3f"),
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
    'yearsActive': None, 'bestQualifyingTime_sec': None, 'resultsDriverId': None,
    # Add to predicted_dnf_position_columns_to_display
    'PredictedDNFProbabilityStd': st.column_config.NumberColumn("Rookie DNF Uncertainty (Std)", format="%.3f"),
}


current_year = datetime.datetime.now().year
raceNoEarlierThan = current_year - 10

# start = time.time()

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
    practices = practices[practices['Driver'] != 'ERROR']  # Remove rows where Driver is 'ERROR'
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
    'trackRace': st.column_config.NumberColumn("Track Race", format="%.3f"),
    'avgLapPace': st.column_config.NumberColumn("Avg. Lap Pace", format="%.3f"),
    'finishingTime': st.column_config.NumberColumn("Finishing Time", format="%.3f")
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
    # Read the header only to get all column names
    all_columns = pd.read_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), sep='\t', nrows=0).columns.tolist()
    selected_columns = ['grandPrixYear', 'grandPrixName', 'resultsDriverName', 'resultsPodium', 'resultsTop5', 'resultsTop10', 'constructorName',  'resultsStartingGridPositionNumber', 'resultsFinalPositionNumber', 
    'positionsGained', 'short_date', 'raceId_results', 'grandPrixRaceId', 'DNF', 'averagePracticePosition', 'lastFPPositionNumber', 'resultsQualificationPositionNumber', 'q1End', 'q2End', 'q3Top10', 'resultsDriverId', 
    'grandPrixLaps', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions', 'turns', 'resultsReasonRetired', 'constructorId_results', 
    'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalPolePositions', 'activeDriver', 'streetRace', 'trackRace', 'recent_form_3_races', 'recent_form_5_races', #'Points',
           'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 'bestQualifyingTime_sec', 'yearsActive', 'driverDNFCount', 'driverDNFAvg',
           'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec', 'LapTime_sec', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'avgLapPace', 'finishingTime', 'constructor_recent_form_3_races', 'constructor_recent_form_5_races',
           'CleanAirAvg_FP1', 'DirtyAirAvg_FP1', 'Delta_FP1', 'CleanAirAvg_FP2', 'DirtyAirAvg_FP2', 'Delta_FP2', 'CleanAirAvg_FP3', 'DirtyAirAvg_FP3','Delta_FP3', 'SafetyCarStatus', 'delta_lap_2', 'delta_lap_5', 'delta_lap_10', 'delta_lap_15', 'delta_lap_20',
            'pit_lane_time_constant', 'pit_stop_delta', 'engineManufacturerId', 'delta_from_race_avg', 'driverAge', 'finishing_position_std_driver', 'finishing_position_std_constructor',
            'delta_lap_2_historical', 'delta_lap_5_historical', 'delta_lap_10_historical', 'delta_lap_15_historical', 'delta_lap_20_historical', 'driver_positionsGained_5_races', 'driver_dnf_rate_5_races',
            'avg_final_position_per_track', 'last_final_position_per_track','avg_final_position_per_track_constructor', 'last_final_position_per_track_constructor',  'qualifying_gap_to_pole',
            'practice_position_improvement_1P_2P', 'practice_position_improvement_2P_3P', 'practice_position_improvement_1P_3P', 'practice_time_improvement_1T_2T', 'practice_time_improvement_time_time', 'practice_time_improvement_2T_3T', 'practice_time_improvement_1T_3T',
            'driverFastestPracticeLap_sec', 'BestConstructorPracticeLap_sec', 'teammate_practice_delta', 'teammate_qual_delta', 'best_qual_time', 
              'qualifying_consistency_std',
            'driver_starting_position_3_races', 'driver_starting_position_5_races', 'abbreviation', 
                                       'qualPos_x_last_practicePos', 'qualPos_x_avg_practicePos', 'recent_form_median_3_races','recent_form_median_5_races', 
                                       'recent_form_best_3_races', 'recent_form_worst_3_races', 'recent_dnf_rate_3_races', 'recent_positions_gained_3_races',
             'driver_positionsGained_3_races', 'qual_vs_track_avg', 'constructor_avg_practice_position', 'practice_position_std', 'recent_vs_season',
            'practice_improvement', 'qual_x_constructor_wins', 'practice_improvement_x_qual',  'grid_penalty', 'grid_penalty_x_constructor', 'recent_form_x_qual', 'practice_std_x_qual',
            'driver_rank_x_constructor_rank', 'grid_x_constructor_rank','driver_rank_x_constructor_rank', 'practice_improvement_x_qual', 'qual_gap_to_teammate', 'practice_gap_to_teammate',
         'recent_form_ratio', 'constructor_form_ratio','total_experience','podium_potential','street_experience','track_experience',
         'fp1_lap_delta_vs_best', 'grid_x_avg_pit_time', 'pit_count_x_pit_delta', 'pit_stop_rate', 'last_race_vs_track_avg',
         'race_pace_vs_median', 'top_speed_rank', 'positions_gained_first_lap_pct',  'power_to_corner_ratio', 'historical_avgLapPace',
         'practice_x_safetycar', 'pit_delta_x_driver_age', 'constructor_points_x_grid', 'dnf_rate_x_practice_std', 'constructor_recent_x_track_exp', 'driver_rank_x_years_active', 
                                         'top_speed_x_turns', 'grid_penalty_x_constructor_rank', 'average_practice_x_driver_podiums',
            'practice_improvement_vs_field', 'constructor_win_rate_3y', 'driver_podium_rate_3y', 'practice_consistency_std', 
                                         'constructor_podium_ratio','practice_to_qualifying_delta', 'track_familiarity', 
                                        'qualifying_position_percentile',   
                                         'recent_podium_streak', 'grid_position_percentile', 'driver_age_squared', 'constructor_recent_win_streak', 'practice_improvement_rate', 'driver_constructor_synergy',
                                        'qual_to_final_delta_5yr', 'qual_to_final_delta_3yr', 'overtake_potential_3yr', 'overtake_potential_5yr',
                                        'driver_avg_qual_pos_at_track','constructor_avg_qual_pos_at_track','driver_avg_grid_pos_at_track','constructor_avg_grid_pos_at_track','driver_avg_practice_pos_at_track',
                                         'constructor_avg_practice_pos_at_track','driver_qual_improvement_3r','constructor_qual_improvement_3r','driver_practice_improvement_3r','constructor_practice_improvement_3r',
                                         'driver_teammate_qual_gap_3r','driver_teammate_practice_gap_3r','driver_street_qual_avg','driver_track_qual_avg','driver_street_practice_avg','driver_track_practice_avg',
                                         'driver_high_wind_qual_avg','driver_high_wind_practice_avg','driver_high_humidity_qual_avg','driver_high_humidity_practice_avg','driver_wet_qual_avg','driver_wet_practice_avg',
                                         'driver_safetycar_qual_avg','driver_safetycar_practice_avg',
                                         'driver_constructor_id','races_with_constructor','is_first_season_with_constructor','driver_constructor_avg_final_position','driver_constructor_avg_qual_position','driver_constructor_podium_rate',
                                         'constructor_dnf_rate_3_races', 'constructor_dnf_rate_5_races', 'recent_dnf_rate_5_races', 'historical_race_pace_vs_median',

                                         'practice_to_qual_improvement_rate','practice_consistency_vs_teammate','qual_vs_constructor_avg_at_track','fp1_lap_time_delta_to_best','q3_lap_time_delta_to_pole',
                                         'fp3_position_percentile','qualifying_position_percentile','constructor_practice_improvement_rate','practice_qual_consistency_5r','track_fp1_fp3_improvement',
                                         'teammate_practice_delta_at_track','constructor_qual_consistency_5r','practice_vs_track_median','qual_vs_track_median','practice_lap_time_improvement_rate',
                                         'practice_improvement_vs_field_avg','qual_improvement_vs_field_avg','practice_to_qual_position_delta','constructor_podium_rate_at_track','driver_podium_rate_at_track',
                                         'fp3_vs_constructor_avg','qual_vs_constructor_avg','practice_lap_time_consistency','qual_lap_time_consistency','practice_improvement_vs_teammate','qual_improvement_vs_teammate',
                                         'practice_vs_best_at_track','qual_vs_best_at_track','practice_vs_worst_at_track','qual_vs_worst_at_track',

                                         'practice_position_percentile_vs_constructor','qualifying_position_percentile_vs_constructor','practice_lap_time_delta_to_constructor_best','qualifying_lap_time_delta_to_constructor_best',
                                         'practice_position_vs_teammate_historical','qualifying_position_vs_teammate_historical','practice_improvement_vs_constructor_historical','qualifying_improvement_vs_constructor_historical',
                                        'practice_consistency_vs_constructor_historical','qualifying_consistency_vs_constructor_historical',
                                        'practice_position_vs_field_best_at_track','qualifying_position_vs_field_best_at_track','practice_position_vs_field_worst_at_track',
                                        'qualifying_position_vs_field_worst_at_track', 'practice_position_vs_field_median_at_track','qualifying_position_vs_field_median_at_track',
                                        'practice_to_qualifying_delta_vs_constructor_historical',
                                        'practice_position_vs_constructor_best_at_track','qualifying_position_vs_constructor_best_at_track',
                                        'practice_position_vs_constructor_worst_at_track','qualifying_position_vs_constructor_worst_at_track',
                                        'practice_position_vs_constructor_median_at_track','qualifying_position_vs_constructor_median_at_track',
                                        'practice_lap_time_consistency_vs_field','qualifying_lap_time_consistency_vs_field',
                                        'practice_position_vs_constructor_recent_form','qualifying_position_vs_constructor_recent_form','practice_position_vs_field_recent_form',
                                        'qualifying_position_vs_field_recent_form', 'currentRookie', 'driver_constructor_id']
    bin_columns = [col for col in all_columns if col.endswith('_bin')]
    usecols = selected_columns + bin_columns
        #    ], dtype={'resultsStartingGridPositionNumber': 'Float64', 'resultsFinalPositionNumber': 'Float64', 'positionsGained': 'Int64', 'averagePracticePosition': 'Float64', 'lastFPPositionNumber': 'Float64', 'resultsQualificationPositionNumber': 'Int64'})
    
    fullResults = pd.read_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), sep='\t', nrows=nrows, usecols=usecols)

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
    fullResults = pd.merge(fullResults, qualifying, left_on=['raceId_results', 'resultsDriverId'], right_on=['raceId', 'driverId'], how='left', suffixes=['_results_with_qualifying', '_qualifying'])
    fullResults.drop_duplicates(subset=['grandPrixYear', 'grandPrixName', 'resultsDriverName'], inplace=True)

    return fullResults

data = load_data(10000)

# Check for duplicate columns and remove them
dupes = [col for col in data.columns if data.columns.tolist().count(col) > 1]
if dupes:
    st.warning(f"Duplicate columns found in your data: {dupes}")
    data = data.loc[:, ~data.columns.duplicated()]

if 'constructorName_results_with_qualifying' in data.columns:
    data.rename(columns={'constructorName_results_with_qualifying': 'constructorName'}, inplace=True)
elif 'constructorName_qualifying' in data.columns:
    data.rename(columns={'constructorName_qualifying': 'constructorName'}, inplace=True)

if 'best_qual_time_results_with_qualifying' in data.columns:
    data.rename(columns={'best_qual_time_results_with_qualifying': 'best_qual_time'}, inplace=True)
elif 'best_qual_time_qualifying' in data.columns:
    data.rename(columns={'best_qual_time_qualifying': 'best_qual_time'}, inplace=True)  

if 'teammate_qual_delta_results_with_qualifying' in data.columns:
    data.rename(columns={'teammate_qual_delta_results_with_qualifying': 'teammate_qual_delta'}, inplace=True)
elif 'teammate_qual_delta_qualifying' in data.columns:
    data.rename(columns={'teammate_qual_delta_qualifying': 'teammate_qual_delta'}, inplace=True)


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
# data['Points'] = data['Points_results_with_qualifying'].astype('Int64')
data['driverRank'] = data['driverRank'].astype('Int64')
if 'bestQualifyingTime_sec' in data.columns:
    data['bestQualifyingTime_sec'] = data['bestQualifyingTime_sec'].astype('Float64')
else:
    st.warning("'bestQualifyingTime_sec' column not found in data.")
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
data['driverAge'] = data['driverAge'].astype('Int64')
data['delta_from_race_avg'] = data['delta_from_race_avg'].astype('Float64')
data['driverAge'] = data['driverAge'].astype('Int64')


column_names = data.columns.tolist()

## do not create filters for any field in this list
## could be to avoid duplicates or not useful for filtering
exclusionList = ['grandPrixRaceId', 'raceId_results',  'constructorId', 'driverId', 'resultsDriverId', 'HeadshotUrl', 'DriverId', 'firstName', 'lastName',
                 'raceId', 'id', 'id_grandPrix', 'id_schedule', 'bestQualifyingTime_sec', 'TeamName', 'circuitId', 'grandPrixRaceId', 'grandPrixId', 'Abbreviation',
                'driverId_driver_standings', 'constructorId_results', 'driverId_results', 'driverId_driver_standings', 'TeamId', 'TeamColor', 'BroadcastName',
                'driverName', 'driverId_driver_standings', 'countryId', 'name', 'fullName', 'points', 'abbreviation', 'shortName', 'id', 'constructorId_results', 'driverId_results',
                'nationalityCountryId', 'secondNationalityCountryId', 'countryOfBirthCountryId', 'placeOfBirth', 'dateOfDeath', 'dateOfBirth', 'gender', 'permanentNumber',
                 'Q1', 'Q2', 'Q3', 'Time', 'PitOutTime', 'PitInTime', 'PitStopTime_sec', 'PitStopTime_mph', 'PitStopTime_mph_avg', 'PitStopTime_sec_avg', 'id_races',
                 'DriverNumber', 'FirstName', 'LastName', 'FullName', 'CountryCode', 'Position', 'ClassifiedPosition', 'GridPosition', 'Status', 'driverDNFCount', 'driverDNFAvg', 'recent_form',
                 'driverNumber', 'Round', 'Year', 'Event', 'totalDriverOfTheDay', 'totalGrandSlams', 'finishingTime', 'average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation',
                  'recent_form_3_races', 'recent_form_5_races', 'constructor_recent_form_3_races', 'constructor_recent_form_5_races','CleanAirAvg_FP1', 'DirtyAirAvg_FP1', 'Delta_FP1', 
                  'CleanAirAvg_FP2', 'DirtyAirAvg_FP2', 'Delta_FP2', 'CleanAirAvg_FP3', 'DirtyAirAvg_FP3','Delta_FP3', 'SafetyCarStatus', 'finishing_position_std_driver', 'finishing_position_std_constructor'
                  'numberOfStops', 'averageStopTime', 'totalStopTime', 'pit_lane_time_constant', 'pit_stop_delta', 'engineManufacturerId',
                  'avg_final_position_per_track', 'last_final_position_per_track', 'avg_final_position_per_track_constructor', 'last_final_position_per_track_constructor',
        'qualifying_gap_to_pole', 'practice_position_improvement_1P_2P',  'practice_position_improvement_1P_3P', 'practice_time_improvement_1T_2T', 'practice_time_improvement_time_time',
        'practice_time_improvement_2T_3T', 'practice_time_improvement_1T_3T', 'driverId_drivers.4', 'abbreviation_drivers.4', 'name_drivers.4', 'firstName_drivers.4', 'lastName_drivers.4', 'driverId_drivers.5', 
        'abbreviation_drivers.5', 'name_drivers.5', 'firstName_drivers.5', 'lastName_drivers.5', 'driverNumber_drivers.4', 'driverId_drivers.6', 'abbreviation_drivers.6', 'name_drivers.6', 
        'firstName_drivers.6', 'lastName_drivers.6', 'driverNumber_drivers.5', 'driverId_drivers.7', 'abbreviation_drivers.7', 'name_drivers.7', 'firstName_drivers.7', 'lastName_drivers.7', 
        'driverNumber_drivers.6',  'delta_lap_2', 'delta_lap_5', 'delta_lap_10', 'delta_lap_15', 'delta_lap_20', 'delta_lap_2_historical', 'delta_lap_5_historical', 'delta_lap_10_historical', 'delta_lap_15_historical', 
        'delta_lap_20_historical', 'driver_positionsGained_5_races', 'driver_dnf_rate_5_races', 'practice_position_improvement_2P_3P',  'finishing_position_std_constructor', 'avgLapPace', 'Laps',
        'driver_starting_position_3_races', 'driver_starting_position_5_races', 'q1_pos', 'q2_pos', 'q3_pos', 
         'delta_from_race_avg', 'driverAge', 'driver_positionsGained_3_races', 'teammate_practice_delta', 'teammate_qual_delta', 'best_qual_time', 'qualifying_consistency_std', 'qual_vs_track_avg', 
         'constructor_avg_practice_position', 'practice_position_std', 'recent_vs_season', 'practice_improvement', 'qual_x_constructor_wins', 'practice_improvement_x_qual', 'grid_penalty', 'grid_penalty_x_constructor', 'recent_form_x_qual', 'practice_std_x_qual', 
                                       'qualPos_x_last_practicePos', 'qualPos_x_avg_practicePos', 'recent_form_median_3_races','recent_form_median_5_races', 'recent_form_best_3_races', 'recent_form_worst_3_races', 'recent_dnf_rate_3_races', 'recent_positions_gained_3_races',
                                       'fp1_lap_delta_vs_best', 'grid_x_avg_pit_time', 'pit_count_x_pit_delta', 'pit_stop_rate', 'last_race_vs_track_avg',
                                       'race_pace_vs_median', 'top_speed_rank', 'positions_gained_first_lap_pct',  'power_to_corner_ratio',
         'driver_rank_x_constructor_rank', 'grid_x_constructor_rank', 'qual_gap_to_teammate', 'practice_gap_to_teammate', 'recent_form_ratio', 'constructor_form_ratio', 'total_experience', 'podium_potential', 
         'street_experience', 'track_experience', 'historical_avgLapPace', 'practice_x_safetycar', 'pit_delta_x_driver_age', 'constructor_points_x_grid', 'dnf_rate_x_practice_std', 'constructor_recent_x_track_exp', 
         'driver_rank_x_years_active', 'top_speed_x_turns', 'grid_penalty_x_constructor_rank', 'average_practice_x_driver_podiums', 'practice_improvement_vs_field', 'constructor_win_rate_3y', 'driver_podium_rate_3y', 
         'practice_consistency_std', 'constructor_podium_ratio', 'practice_to_qualifying_delta', 'track_familiarity', 'qualifying_position_percentile', 'recent_podium_streak', 'grid_position_percentile', 'driver_age_squared', 
         'constructor_recent_win_streak', 'practice_improvement_rate', 'driver_constructor_synergy', 'qual_to_final_delta_5yr', 'qual_to_final_delta_3yr', 'overtake_potential_3yr', 'overtake_potential_5yr', 
         'driver_avg_qual_pos_at_track','constructor_avg_qual_pos_at_track','driver_avg_grid_pos_at_track','constructor_avg_grid_pos_at_track','driver_avg_practice_pos_at_track',
                                         'constructor_avg_practice_pos_at_track','driver_qual_improvement_3r','constructor_qual_improvement_3r','driver_practice_improvement_3r','constructor_practice_improvement_3r',
                                         'driver_teammate_qual_gap_3r','driver_teammate_practice_gap_3r','driver_street_qual_avg','driver_track_qual_avg','driver_street_practice_avg','driver_track_practice_avg',
                                         'driver_high_wind_qual_avg','driver_high_wind_practice_avg','driver_high_humidity_qual_avg','driver_high_humidity_practice_avg','driver_wet_qual_avg','driver_wet_practice_avg',
                                         'driver_safetycar_qual_avg','driver_safetycar_practice_avg', 'historical_race_pace_vs_median',
                                         'races_with_constructor','is_first_season_with_constructor','driver_constructor_avg_final_position','driver_constructor_avg_qual_position','driver_constructor_podium_rate',
                                         'constructor_dnf_rate_3_races', 'constructor_dnf_rate_5_races', 'recent_dnf_rate_5_races',  
                                         'practice_to_qual_improvement_rate','practice_consistency_vs_teammate','qual_vs_constructor_avg_at_track','fp1_lap_time_delta_to_best','q3_lap_time_delta_to_pole',
                                         'fp3_position_percentile','qualifying_position_percentile','constructor_practice_improvement_rate','practice_qual_consistency_5r','track_fp1_fp3_improvement',
                                         'teammate_practice_delta_at_track','constructor_qual_consistency_5r','practice_vs_track_median','qual_vs_track_median','practice_lap_time_improvement_rate',
                                         'practice_improvement_vs_field_avg','qual_improvement_vs_field_avg','practice_to_qual_position_delta','constructor_podium_rate_at_track','driver_podium_rate_at_track',
                                         'fp3_vs_constructor_avg','qual_vs_constructor_avg','practice_lap_time_consistency','qual_lap_time_consistency','practice_improvement_vs_teammate','qual_improvement_vs_teammate',
                                         'practice_vs_best_at_track','qual_vs_best_at_track','practice_vs_worst_at_track','qual_vs_worst_at_track',

                                         'practice_position_percentile_vs_constructor','qualifying_position_percentile_vs_constructor','practice_lap_time_delta_to_constructor_best','qualifying_lap_time_delta_to_constructor_best',
                                         'practice_position_vs_teammate_historical','qualifying_position_vs_teammate_historical_bin','practice_improvement_vs_constructor_historical','qualifying_improvement_vs_constructor_historical',
                                        'practice_consistency_vs_constructor_historical','qualifying_consistency_vs_constructor_historical',
                                        'practice_position_vs_field_best_at_track','qualifying_position_vs_field_best_at_track','practice_position_vs_field_worst_at_track',
                                        'qualifying_position_vs_field_worst_at_track', 'practice_position_vs_field_median_at_track','qualifying_position_vs_field_median_at_track',
                                        'practice_to_qualifying_delta_vs_constructor_historical',
                                        'practice_position_vs_constructor_best_at_track','qualifying_position_vs_constructor_best_at_track',
                                        'practice_position_vs_constructor_worst_at_track','qualifying_position_vs_constructor_worst_at_track',
                                        'practice_position_vs_constructor_median_at_track','qualifying_position_vs_constructor_median_at_track',
                                        'practice_lap_time_consistency_vs_field','qualifying_lap_time_consistency_vs_field',
                                        'practice_position_vs_constructor_recent_form','qualifying_position_vs_constructor_recent_form','practice_position_vs_field_recent_form',
                                        'qualifying_position_vs_field_recent_form'                                     
        ]

suffixes_to_exclude = ('_x', '_y', '_qualifying', '_results_with_qualifying', '_drivers', '_mph', '_sec', '.1', '.2', '.3', '_bin')
auto_exclusions = [col for col in column_names if col.endswith(suffixes_to_exclude)]
exclusionList = exclusionList + auto_exclusions

# If errant/extra columns appear on the left in filters, the below analysis will point them out.

# st.write(f"Exclusion List: {exclusionList}")

# remaining_columns = [col for col in column_names if col not in exclusionList]
# st.write(f"Remaining Columns: {remaining_columns}")

# column_names.sort()


# from 9/19/2025
# def get_features_and_target(data):
#     features = ['recent_positions_gained_3_races', 'constructorName', 'average_temp', 'constructor_recent_win_streak', 'practice_time_improvement_1T_3T', 'constructorTotalRaceWins', 
#                 'average_humidity', 'SpeedST_mph', 'recent_form_median_5_races', 'driverDNFAvg', 'qual_x_constructor_wins', 'teammate_qual_delta', 'track_experience', 'driverAge', 
#                 'resultsDriverName', 'average_wind_speed', 'grid_x_constructor_rank', 'qualPos_x_avg_practicePos', 'trackRace', 'driverTotalRaceStarts', 
#                 'SpeedI2_mph', 'averageStopTime', 'driver_rank_x_constructor_rank', 'Delta_FP1', 'pit_stop_rate', 'qualifying_consistency_std', 'grid_x_avg_pit_time', 
#                 'driver_teammate_practice_gap_3r', 'delta_from_race_avg', 'last_final_position_per_track_constructor', 'recent_form_ratio', 'constructor_dnf_rate_5_races', 
#                 'best_s3_sec', 'podium_potential', 'historical_race_pace_vs_median', 'practice_time_improvement_time_time', 'teammate_practice_delta', 'Points', 'driver_podium_rate_3y', 
#                 'turns', 'positions_gained_first_lap_pct', 'driverDNFCount', 'totalChampionshipPoints', 'avg_final_position_per_track', 'driver_street_practice_avg', 'Delta_FP3', 
#                 'driver_avg_grid_pos_at_track', 'CleanAirAvg_FP1', 'practice_position_improvement_2P_3P', 'recent_dnf_rate_3_races', 'qualifying_gap_to_pole', 
#                 'driver_positionsGained_3_races', 'Delta_FP2', 'driver_starting_position_5_races', 'grid_penalty_x_constructor', 'constructor_avg_practice_position', 'SpeedI1_mph', 
#                 'practice_std_x_qual', 'averagePracticePosition', 'grid_penalty', 'driver_track_qual_avg', 'best_theory_lap_sec', 'totalStopTime', 'recent_form_x_qual', 
#                 'CleanAirAvg_FP2', 'best_s2_sec', 'constructor_recent_form_5_races', 'constructorTotalRaceStarts', 'driver_age_squared', 'street_experience', 
#                 'driver_track_practice_avg', 'driverFastestPracticeLap_sec', 'engineManufacturerId', 'is_first_season_with_constructor', 'recent_form_5_races', 
#                 'constructor_podium_ratio', 'driver_starting_position_3_races', 'pit_stop_delta', 'recent_form_median_3_races', 'driver_dnf_rate_5_races', 'driver_high_wind_qual_avg', 'grandPrixName', 'BestConstructorPracticeLap_sec', 
#                 'practice_time_improvement_1T_2T', 'yearsActive', 'SpeedFL_mph', 'driver_constructor_podium_rate', 'practice_improvement_x_qual', 'practice_improvement', 'total_experience', 
#                 'best_qual_time', 'driver_high_wind_practice_avg', 'practice_time_improvement_2T_3T', 'practice_position_improvement_1P_3P', 'driver_avg_qual_pos_at_track', 
#                 'recent_form_best_3_races', 'qual_vs_track_avg', 'constructor_recent_form_3_races', 'streetRace', 'totalPolePositions', 'driverTotalChampionshipWins', 'fp1_lap_delta_vs_best', 
#                 'driver_positionsGained_5_races', 'qualPos_x_last_practicePos', 'race_pace_vs_median',

#                 'practice_to_qual_improvement_rate','practice_consistency_vs_teammate','qual_vs_constructor_avg_at_track','fp1_lap_time_delta_to_best','q3_lap_time_delta_to_pole',
#                                          'fp3_position_percentile','qualifying_position_percentile','constructor_practice_improvement_rate','practice_qual_consistency_5r','track_fp1_fp3_improvement',
#                                          'teammate_practice_delta_at_track','constructor_qual_consistency_5r','practice_vs_track_median','qual_vs_track_median','practice_lap_time_improvement_rate',
#                                          'practice_improvement_vs_field_avg','qual_improvement_vs_field_avg','practice_to_qual_position_delta','constructor_podium_rate_at_track','driver_podium_rate_at_track',
#                                          'fp3_vs_constructor_avg','qual_vs_constructor_avg','practice_lap_time_consistency','qual_lap_time_consistency','practice_improvement_vs_teammate','qual_improvement_vs_teammate',
#                                          'practice_vs_best_at_track','qual_vs_best_at_track','practice_vs_worst_at_track','qual_vs_worst_at_track',
                                         
#                                          'practice_position_percentile_vs_constructor','qualifying_position_percentile_vs_constructor','practice_lap_time_delta_to_constructor_best','qualifying_lap_time_delta_to_constructor_best',
#                                          'practice_position_vs_teammate_historical','qualifying_position_vs_teammate_historical','practice_improvement_vs_constructor_historical','qualifying_improvement_vs_constructor_historical',
#                                         'practice_consistency_vs_constructor_historical','qualifying_consistency_vs_constructor_historical',
#                                         'practice_position_vs_field_best_at_track','qualifying_position_vs_field_best_at_track','practice_position_vs_field_worst_at_track',
#                                         'qualifying_position_vs_field_worst_at_track', 'practice_position_vs_field_median_at_track','qualifying_position_vs_field_median_at_track',
#                                         'practice_to_qualifying_delta_vs_constructor_historical',
#                                         'practice_position_vs_constructor_best_at_track','qualifying_position_vs_constructor_best_at_track',
#                                         'practice_position_vs_constructor_worst_at_track','qualifying_position_vs_constructor_worst_at_track',
#                                         'practice_position_vs_constructor_median_at_track','qualifying_position_vs_constructor_median_at_track',
#                                         'practice_lap_time_consistency_vs_field','qualifying_lap_time_consistency_vs_field',
#                                         'practice_position_vs_constructor_recent_form','qualifying_position_vs_constructor_recent_form','practice_position_vs_field_recent_form',
#                                         'qualifying_position_vs_field_recent_form', 'currentRookie'

#             ]
#     target = 'resultsFinalPositionNumber'
#     return data[features], data[target]

# all of the non-leaky fields from the fullResults dataset (9/19/2025)
def get_features_and_target(data):
    features = [ 'constructorName', 'grandPrixName', 'resultsDriverName',  'resultsStartingGridPositionNumber',  
    'averagePracticePosition', 'lastFPPositionNumber', 'resultsQualificationPositionNumber', 'q1End', 'q2End', 'q3Top10',  
    'grandPrixLaps', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions', 'turns', 
    'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalPolePositions', 'activeDriver', 'streetRace', 'trackRace', 'recent_form_3_races', 'recent_form_5_races', #'Points',
           'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 'bestQualifyingTime_sec_bin', 'yearsActive', 'driverDNFCount', 'driverDNFAvg',
           'best_s1_sec_bin', 'best_s2_sec_bin', 'best_s3_sec_bin', 'best_theory_lap_sec_bin', 'LapTime_sec_bin', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph',   'constructor_recent_form_3_races', 'constructor_recent_form_5_races',
           'CleanAirAvg_FP1_bin', 'DirtyAirAvg_FP1_bin', 'Delta_FP1_bin', 'CleanAirAvg_FP2_bin', 'DirtyAirAvg_FP2_bin', 'Delta_FP2_bin', 'CleanAirAvg_FP3_bin', 'DirtyAirAvg_FP3_bin','Delta_FP3_bin', #'delta_lap_2', 'delta_lap_5', 'delta_lap_10', 'delta_lap_15', 'delta_lap_20',
             'engineManufacturerId', 'delta_from_race_avg_bin', 'driverAge', 'finishing_position_std_driver', 'finishing_position_std_constructor',
            'delta_lap_2_historical', 'delta_lap_5_historical', 'delta_lap_10_historical', 'delta_lap_15_historical', 'delta_lap_20_historical', 'driver_positionsGained_5_races', 'driver_dnf_rate_5_races',
            'avg_final_position_per_track', 'last_final_position_per_track','avg_final_position_per_track_constructor', 
            'practice_position_improvement_1P_2P', 'practice_position_improvement_2P_3P', 'practice_position_improvement_1P_3P', 'practice_time_improvement_1T_2T_bin', 'practice_time_improvement_time_time_bin', 'practice_time_improvement_2T_3T_bin', 'practice_time_improvement_1T_3T_bin',
            'driverFastestPracticeLap_sec_bin', 'BestConstructorPracticeLap_sec_bin', 'teammate_practice_delta_bin', 'teammate_qual_delta_bin', 'best_qual_time_bin', 
            'last_final_position_per_track_constructor',  'qualifying_gap_to_pole', 'qualifying_consistency_std',
            'driver_starting_position_3_races', 'driver_starting_position_5_races', 
                                       'qualPos_x_last_practicePos', 'qualPos_x_avg_practicePos', 'recent_form_median_3_races','recent_form_median_5_races', 
                                       'recent_form_best_3_races', 'recent_form_worst_3_races', 'recent_dnf_rate_3_races', 'recent_positions_gained_3_races',
           'driver_positionsGained_3_races', 'qual_vs_track_avg_bin', 'constructor_avg_practice_position', 'practice_position_std', 'recent_vs_season',
            'practice_improvement', 'qual_x_constructor_wins', 'practice_improvement_x_qual',  'grid_penalty', 'grid_penalty_x_constructor', 'recent_form_x_qual', 'practice_std_x_qual',
            'grid_x_constructor_rank','driver_rank_x_constructor_rank', 'qual_gap_to_teammate_bin', 'practice_gap_to_teammate_bin',
         'recent_form_ratio', 'constructor_form_ratio','total_experience','podium_potential','street_experience','track_experience',
         'fp1_lap_delta_vs_best_bin', 'grid_x_avg_pit_time_bin',  'last_race_vs_track_avg_bin',
          'top_speed_rank', 'positions_gained_first_lap_pct',  'power_to_corner_ratio', 'historical_avgLapPace_bin',
         'practice_x_safetycar', 'pit_delta_x_driver_age_bin', 'constructor_points_x_grid', 'dnf_rate_x_practice_std_bin', 'constructor_recent_x_track_exp', 'driver_rank_x_years_active', 
                                         'top_speed_x_turns', 'grid_penalty_x_constructor_rank', 'average_practice_x_driver_podiums',
            'practice_improvement_vs_field_bin', 'constructor_win_rate_3y', 'driver_podium_rate_3y', 'practice_consistency_std', 
                                         'constructor_podium_ratio','practice_to_qualifying_delta', 'track_familiarity', 
                                        'qualifying_position_percentile',   
                                         'recent_podium_streak', 'grid_position_percentile', 'driver_age_squared', 'constructor_recent_win_streak', 'practice_improvement_rate', 'driver_constructor_synergy',
                                        'qual_to_final_delta_5yr', 'qual_to_final_delta_3yr', 'overtake_potential_3yr', 'overtake_potential_5yr',
                                        'driver_avg_qual_pos_at_track','constructor_avg_qual_pos_at_track','driver_avg_grid_pos_at_track','constructor_avg_grid_pos_at_track','driver_avg_practice_pos_at_track',
                                         'constructor_avg_practice_pos_at_track','driver_qual_improvement_3r','constructor_qual_improvement_3r','driver_practice_improvement_3r','constructor_practice_improvement_3r',
                                         'driver_teammate_qual_gap_3r','driver_teammate_practice_gap_3r_bin','driver_street_qual_avg','driver_track_qual_avg','driver_street_practice_avg','driver_track_practice_avg',
                                         'driver_high_wind_qual_avg','driver_high_wind_practice_avg','driver_high_humidity_qual_avg','driver_high_humidity_practice_avg','driver_wet_qual_avg','driver_wet_practice_avg',
                                         'driver_safetycar_qual_avg','driver_safetycar_practice_avg',
                                         'races_with_constructor','is_first_season_with_constructor','driver_constructor_avg_final_position','driver_constructor_avg_qual_position','driver_constructor_podium_rate',
                                         'constructor_dnf_rate_3_races', 'constructor_dnf_rate_5_races', 'recent_dnf_rate_5_races', 'historical_race_pace_vs_median_bin',

                                         'practice_to_qual_improvement_rate','practice_consistency_vs_teammate_bin','qual_vs_constructor_avg_at_track','fp1_lap_time_delta_to_best_bin','q3_lap_time_delta_to_pole',
                                         'fp3_position_percentile','constructor_practice_improvement_rate','practice_qual_consistency_5r_bin','track_fp1_fp3_improvement',
                                         'teammate_practice_delta_at_track_bin','constructor_qual_consistency_5r_bin','practice_vs_track_median','qual_vs_track_median','practice_lap_time_improvement_rate_bin',
                                         'practice_improvement_vs_field_avg','qual_improvement_vs_field_avg','practice_to_qual_position_delta','constructor_podium_rate_at_track','driver_podium_rate_at_track',
                                         'fp3_vs_constructor_avg','qual_vs_constructor_avg','practice_lap_time_consistency_bin','qual_lap_time_consistency_bin','practice_improvement_vs_teammate','qual_improvement_vs_teammate',
                                         'practice_vs_best_at_track','qual_vs_best_at_track','practice_vs_worst_at_track','qual_vs_worst_at_track',

                                         'practice_position_percentile_vs_constructor_bin','qualifying_position_percentile_vs_constructor_bin','practice_lap_time_delta_to_constructor_best_bin','qualifying_lap_time_delta_to_constructor_best_bin',
                                         'practice_position_vs_teammate_historical_bin','qualifying_position_vs_teammate_historical_bin','practice_improvement_vs_constructor_historical_bin','qualifying_improvement_vs_constructor_historical_bin',
                                        'practice_consistency_vs_constructor_historical_bin','qualifying_consistency_vs_constructor_historical_bin',
                                        'practice_position_vs_field_best_at_track','qualifying_position_vs_field_best_at_track','practice_position_vs_field_worst_at_track',
                                        'qualifying_position_vs_field_worst_at_track', 'practice_position_vs_field_median_at_track','qualifying_position_vs_field_median_at_track',
                                        'practice_to_qualifying_delta_vs_constructor_historical',
                                        'practice_position_vs_constructor_best_at_track','qualifying_position_vs_constructor_best_at_track',
                                        'practice_position_vs_constructor_worst_at_track','qualifying_position_vs_constructor_worst_at_track',
                                        'practice_position_vs_constructor_median_at_track','qualifying_position_vs_constructor_median_at_track',
                                        'practice_lap_time_consistency_vs_field_bin','qualifying_lap_time_consistency_vs_field_bin',
                                        'practice_position_vs_constructor_recent_form','qualifying_position_vs_constructor_recent_form','practice_position_vs_field_recent_form_bin',
                                        'qualifying_position_vs_field_recent_form', 'currentRookie'

            ]
    dupes = [col for col in features if features.count(col) > 1]
    if dupes:
        st.warning(f"Duplicate features in your feature list: {dupes}")
        features = list(dict.fromkeys(features))  # Remove duplicates, keep order
    target = 'resultsFinalPositionNumber'
    return data[features], data[target]

features, _ = get_features_and_target(data)
missing = [col for col in features.columns if col not in data.columns]
if missing:
    st.write(f"The following feature columns are missing from your data: {missing}")
    st.stop()

def get_preprocessor_position():
    categorical_features = [
        # 'grandPrixName', 
        'engineManufacturerId', 
        'constructorName', 
        # 'resultsDriverName', 
        # 'driver_constructor_id'
        ]
    numerical_features = [ 

                 'resultsStartingGridPositionNumber',  'averagePracticePosition', 'lastFPPositionNumber', 'resultsQualificationPositionNumber',  'q1End', 'q2End', 'q3Top10',
    'grandPrixLaps', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions', 'turns', 
    'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalPolePositions', 'activeDriver', 'streetRace', 'trackRace', 'recent_form_3_races', 'recent_form_5_races', #'Points',
           'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 'bestQualifyingTime_sec_bin', 'yearsActive', 'driverDNFCount', 'driverDNFAvg',
           'best_s1_sec_bin', 'best_s2_sec_bin', 'best_s3_sec_bin', 'best_theory_lap_sec_bin', 'LapTime_sec_bin', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'constructor_recent_form_3_races', 'constructor_recent_form_5_races',
           'CleanAirAvg_FP1_bin', 'DirtyAirAvg_FP1_bin', 'Delta_FP1_bin', 'CleanAirAvg_FP2_bin', 'DirtyAirAvg_FP2_bin', 'Delta_FP2_bin', 'CleanAirAvg_FP3_bin', 'DirtyAirAvg_FP3_bin','Delta_FP3_bin', #'delta_lap_2', 'delta_lap_5', 'delta_lap_10', 'delta_lap_15', 'delta_lap_20',
             'delta_from_race_avg_bin', 'driverAge', 'finishing_position_std_driver', 'finishing_position_std_constructor',
            'delta_lap_2_historical', 'delta_lap_5_historical', 'delta_lap_10_historical', 'delta_lap_15_historical', 'delta_lap_20_historical', 'driver_positionsGained_5_races', 'driver_dnf_rate_5_races',
            'avg_final_position_per_track', 'last_final_position_per_track','avg_final_position_per_track_constructor', 'last_final_position_per_track_constructor',  'qualifying_gap_to_pole',
            'practice_position_improvement_1P_2P', 'practice_position_improvement_2P_3P', 'practice_position_improvement_1P_3P', 'practice_time_improvement_1T_2T_bin', 'practice_time_improvement_time_time_bin', 'practice_time_improvement_2T_3T_bin', 'practice_time_improvement_1T_3T_bin',
            'driverFastestPracticeLap_sec_bin', 'BestConstructorPracticeLap_sec_bin', 'teammate_practice_delta_bin', 'teammate_qual_delta_bin', 'best_qual_time_bin',  
             'qualifying_consistency_std',
            'driver_starting_position_3_races', 'driver_starting_position_5_races', 
                                       'qualPos_x_last_practicePos', 'qualPos_x_avg_practicePos', 'recent_form_median_3_races','recent_form_median_5_races', 
                                       'recent_form_best_3_races', 'recent_form_worst_3_races', 'recent_dnf_rate_3_races', 'recent_positions_gained_3_races',
             'driver_positionsGained_3_races', 'qual_vs_track_avg_bin', 'constructor_avg_practice_position', 'practice_position_std', 'recent_vs_season',
            'practice_improvement', 'qual_x_constructor_wins', 'practice_improvement_x_qual',  'grid_penalty', 'grid_penalty_x_constructor', 'recent_form_x_qual', 'practice_std_x_qual',
            'driver_rank_x_constructor_rank', 'grid_x_constructor_rank','practice_improvement_x_qual', 'qual_gap_to_teammate_bin', 'practice_gap_to_teammate_bin',
         'recent_form_ratio', 'constructor_form_ratio','total_experience','podium_potential','street_experience','track_experience',
         'fp1_lap_delta_vs_best_bin', 'grid_x_avg_pit_time_bin', 'last_race_vs_track_avg_bin',
          'top_speed_rank', 'positions_gained_first_lap_pct',  'power_to_corner_ratio', 'historical_avgLapPace_bin',
         'practice_x_safetycar', 'pit_delta_x_driver_age_bin', 'constructor_points_x_grid', 'dnf_rate_x_practice_std_bin', 'constructor_recent_x_track_exp', 'driver_rank_x_years_active', 
                                         'top_speed_x_turns', 'grid_penalty_x_constructor_rank', 'average_practice_x_driver_podiums',
            'practice_improvement_vs_field_bin', 'constructor_win_rate_3y', 'driver_podium_rate_3y', 'practice_consistency_std', 
                                         'constructor_podium_ratio','practice_to_qualifying_delta', 'track_familiarity', 
                                         
                                         'recent_podium_streak', 'grid_position_percentile', 'driver_age_squared', 'constructor_recent_win_streak', 'practice_improvement_rate', 'driver_constructor_synergy',
                                        'qual_to_final_delta_5yr', 'qual_to_final_delta_3yr', 'overtake_potential_3yr', 'overtake_potential_5yr',
                                        'driver_avg_qual_pos_at_track','constructor_avg_qual_pos_at_track','driver_avg_grid_pos_at_track','constructor_avg_grid_pos_at_track','driver_avg_practice_pos_at_track',
                                         'constructor_avg_practice_pos_at_track','driver_qual_improvement_3r','constructor_qual_improvement_3r','driver_practice_improvement_3r','constructor_practice_improvement_3r',
                                         'driver_teammate_qual_gap_3r','driver_teammate_practice_gap_3r_bin','driver_street_qual_avg','driver_track_qual_avg','driver_street_practice_avg','driver_track_practice_avg',
                                         'driver_high_wind_qual_avg','driver_high_wind_practice_avg','driver_high_humidity_qual_avg','driver_high_humidity_practice_avg','driver_wet_qual_avg','driver_wet_practice_avg',
                                         'driver_safetycar_qual_avg','driver_safetycar_practice_avg',
                                         'races_with_constructor','is_first_season_with_constructor','driver_constructor_avg_final_position','driver_constructor_avg_qual_position','driver_constructor_podium_rate',
                                         'constructor_dnf_rate_3_races', 'constructor_dnf_rate_5_races', 'recent_dnf_rate_5_races', 'historical_race_pace_vs_median_bin',

                                         'practice_to_qual_improvement_rate','practice_consistency_vs_teammate_bin','qual_vs_constructor_avg_at_track','fp1_lap_time_delta_to_best_bin','q3_lap_time_delta_to_pole',
                                         'fp3_position_percentile','qualifying_position_percentile','constructor_practice_improvement_rate','practice_qual_consistency_5r_bin','track_fp1_fp3_improvement',
                                         'teammate_practice_delta_at_track_bin','constructor_qual_consistency_5r_bin','practice_vs_track_median','qual_vs_track_median','practice_lap_time_improvement_rate_bin',
                                         'practice_improvement_vs_field_avg','qual_improvement_vs_field_avg','practice_to_qual_position_delta','constructor_podium_rate_at_track','driver_podium_rate_at_track',
                                         'fp3_vs_constructor_avg','qual_vs_constructor_avg','practice_lap_time_consistency_bin','qual_lap_time_consistency_bin','practice_improvement_vs_teammate','qual_improvement_vs_teammate',
                                         'practice_vs_best_at_track','qual_vs_best_at_track','practice_vs_worst_at_track','qual_vs_worst_at_track',

                                         'practice_position_percentile_vs_constructor_bin','qualifying_position_percentile_vs_constructor_bin','practice_lap_time_delta_to_constructor_best_bin','qualifying_lap_time_delta_to_constructor_best_bin',
                                         'practice_position_vs_teammate_historical_bin','qualifying_position_vs_teammate_historical_bin','practice_improvement_vs_constructor_historical_bin','qualifying_improvement_vs_constructor_historical_bin',
                                        'practice_consistency_vs_constructor_historical_bin','qualifying_consistency_vs_constructor_historical_bin',
                                        'practice_position_vs_field_best_at_track','qualifying_position_vs_field_best_at_track','practice_position_vs_field_worst_at_track',
                                        'qualifying_position_vs_field_worst_at_track', 'practice_position_vs_field_median_at_track','qualifying_position_vs_field_median_at_track',
                                        'practice_to_qualifying_delta_vs_constructor_historical',
                                        'practice_position_vs_constructor_best_at_track','qualifying_position_vs_constructor_best_at_track',
                                        'practice_position_vs_constructor_worst_at_track','qualifying_position_vs_constructor_worst_at_track',
                                        'practice_position_vs_constructor_median_at_track','qualifying_position_vs_constructor_median_at_track',
                                        'practice_lap_time_consistency_vs_field_bin','qualifying_lap_time_consistency_vs_field_bin',
                                        'practice_position_vs_constructor_recent_form','qualifying_position_vs_constructor_recent_form','practice_position_vs_field_recent_form_bin',
                                        'qualifying_position_vs_field_recent_form', 'currentRookie'
         
          ]

# 9/19/2025
# def get_preprocessor_position():
#     categorical_features = [
#         'grandPrixName', 
#         'engineManufacturerId', 
#         'constructorName', 
#         'resultsDriverName', ]
#     numerical_features = [ 

#                 'currentRookie',
#                 # 'qual_vs_worst_at_track',

#                 'practice_position_percentile_vs_constructor',
#                 'qualifying_position_percentile_vs_constructor',
#                 # 'practice_lap_time_delta_to_constructor_best',
#                 # 'qualifying_lap_time_delta_to_constructor_best',
#                 'practice_position_vs_teammate_historical',
#                 'qualifying_position_vs_teammate_historical',
#                 # 'practice_improvement_vs_constructor_historical',
#                 'qualifying_improvement_vs_constructor_historical',
#                 # 'practice_consistency_vs_constructor_historical',
#                 # 'qualifying_consistency_vs_constructor_historical',
#                 # 'practice_position_vs_field_best_at_track',
#                 # 'qualifying_position_vs_field_best_at_track',
#                 # 'practice_position_vs_field_worst_at_track',
#                 # 'qualifying_position_vs_field_worst_at_track',
#                 # 'practice_position_vs_field_median_at_track',
#                 # 'qualifying_position_vs_field_median_at_track',
#                 # 'practice_to_qualifying_delta_vs_constructor_historical',
#                 # 'practice_position_vs_constructor_best_at_track',
#                 # 'qualifying_position_vs_constructor_best_at_track',
#                 # 'practice_position_vs_constructor_worst_at_track',
#                 # 'qualifying_position_vs_constructor_worst_at_track',
#                 'practice_position_vs_constructor_median_at_track',
#                 # 'qualifying_position_vs_constructor_median_at_track',
#                 # 'practice_lap_time_consistency_vs_field',
#                 # 'qualifying_lap_time_consistency_vs_field',
#                 # 'practice_position_vs_constructor_recent_form',
#                 # 'qualifying_position_vs_constructor_recent_form',
#                 # 'practice_position_vs_field_recent_form',
#                 # 'qualifying_position_vs_field_recent_form',                
                
                
#                 # 'practice_to_qual_improvement_rate',
                
#                 'practice_consistency_vs_teammate',
#                 'qual_vs_constructor_avg_at_track',
               
#                 # 'fp1_lap_time_delta_to_best',
                

#                 # 'q3_lap_time_delta_to_pole',
                

#                 'fp3_position_percentile',
                
#                 # 'qualifying_position_percentile',
#                 # 'constructor_practice_improvement_rate',
                
#                 'practice_qual_consistency_5r',
#                 'track_fp1_fp3_improvement',
                
#                 # 'teammate_practice_delta_at_track',
#                 # 'constructor_qual_consistency_5r',
                
#                 'practice_vs_track_median',
#                 'qual_vs_track_median',
                
#                 'practice_lap_time_improvement_rate',
                
#                 'practice_improvement_vs_field_avg',
#                 'qual_improvement_vs_field_avg',
#                 'practice_to_qual_position_delta',
#                 'constructor_podium_rate_at_track',
#                 'driver_podium_rate_at_track',
#                 'fp3_vs_constructor_avg',
#                 'qual_vs_constructor_avg',
#                 'practice_lap_time_consistency',
#                 'qual_lap_time_consistency',
#                 'practice_improvement_vs_teammate',
#                 'qual_improvement_vs_teammate',
#                 'practice_vs_best_at_track',
#                 'qual_vs_best_at_track',
#                 'practice_vs_worst_at_track',



#                 'recent_positions_gained_3_races',
#                 'average_temp',
#                 # 'constructor_recent_win_streak',
#                 'practice_time_improvement_1T_3T',
#                 # 'constructorTotalRaceWins',
#                 'average_humidity',
#                 'SpeedST_mph',
#                 'recent_form_median_5_races',
#                 'driverDNFAvg',
#                 'qual_x_constructor_wins',
#                 'teammate_qual_delta',
#                 'track_experience',
#                 'driverAge',
#                 'average_wind_speed',
#                 'grid_x_constructor_rank',
#                 'qualPos_x_avg_practicePos',
                
                
#                 # 'trackRace',
#                 'driverTotalRaceStarts',
#                 'SpeedI2_mph',
#                 'averageStopTime',
#                 'driver_rank_x_constructor_rank',
#                 'Delta_FP1',
#                 'pit_stop_rate',
#                 'qualifying_consistency_std',
#                 'grid_x_avg_pit_time',
#                 'driver_teammate_practice_gap_3r',
#                 'delta_from_race_avg',
#                 'last_final_position_per_track_constructor',
#                 'recent_form_ratio',
#                 'constructor_dnf_rate_5_races',
#                 'best_s3_sec',
#                 'podium_potential',
#                 'historical_race_pace_vs_median',
#                 # 'practice_time_improvement_time_time',
#                 'teammate_practice_delta',
#                 'Points',
#                 'driver_podium_rate_3y',
#                 'turns',
#                 'positions_gained_first_lap_pct',
#                 'driverDNFCount',
#                 'totalChampionshipPoints',
#                 'avg_final_position_per_track',
#                 'driver_street_practice_avg',
#                 'Delta_FP3',
#                 'driver_avg_grid_pos_at_track',
#                 'CleanAirAvg_FP1',
#                 'practice_position_improvement_2P_3P',
#                 'recent_dnf_rate_3_races',
#                 # 'qualifying_gap_to_pole',
#                 # 'driver_positionsGained_3_races',
#                 'Delta_FP2',
#                 'driver_starting_position_5_races',
#                 'grid_penalty_x_constructor',
#                 'constructor_avg_practice_position',
#                 # 'SpeedI1_mph',
#                 'practice_std_x_qual',
#                 'averagePracticePosition',
#                 'grid_penalty',
#                 'driver_track_qual_avg',
#                 'best_theory_lap_sec',
#                 'totalStopTime',
#                 'recent_form_x_qual',
#                 'CleanAirAvg_FP2',
#                 'best_s2_sec',
#                 'constructor_recent_form_5_races',
#                 'constructorTotalRaceStarts',
#                 # 'driver_age_squared',
#                 'street_experience',
                
                
#                 'driver_track_practice_avg',
#                 # 'driverFastestPracticeLap_sec',
#                 # 'is_first_season_with_constructor',
#                 'recent_form_5_races',
#                 'constructor_podium_ratio',
#                 'driver_starting_position_3_races',
#                 'pit_stop_delta',
#                 'recent_form_median_3_races',
#                 'driver_dnf_rate_5_races',
#                 'driver_high_wind_qual_avg',
#                 'BestConstructorPracticeLap_sec',
#                 'practice_time_improvement_1T_2T',
#                 'yearsActive',
#                 'SpeedFL_mph',
#                 'driver_constructor_podium_rate',
#                 'practice_improvement_x_qual',
#                 'practice_improvement',
#                 # 'total_experience',
#                 'best_qual_time',
#                 # 'driver_high_wind_practice_avg',
#                 # 'practice_time_improvement_2T_3T',
                
#                 # 'practice_position_improvement_1P_3P',
                

#                 'driver_avg_qual_pos_at_track',
#                 'recent_form_best_3_races',
#                 # 'qual_vs_track_avg',
#                 # 'constructor_recent_form_3_races',
#                 # 'streetRace',
#                 'totalPolePositions',
#                 # 'driverTotalChampionshipWins',
#                 'fp1_lap_delta_vs_best',
#                 'driver_positionsGained_5_races',
#                 # 'qualPos_x_last_practicePos',
#                 'race_pace_vs_median'
         
#           ]


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
    'averagePracticePosition', 'lastFPPositionNumber', 'resultsStartingGridPositionNumber', 'numberOfStops',
    'trackRace', 'streetRace', 'turns', 'average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation', 
    'driverDNFCount', 'driverDNFAvg', 'driver_dnf_rate_5_races', 'recent_dnf_rate_3_races',  'constructor_dnf_rate_3_races', 
    'constructor_dnf_rate_5_races', 'total_experience', 'driverAge',
       
    # Add weather features if available
    ]
    target = 'DNF'
    return data[features], data[target]

def get_preprocessor_dnf():
    categorical_features = ['grandPrixName', 'constructorName', 'resultsDriverName']
    numerical_features = ['driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalChampionshipWins',
    'driverTotalRaceWins', 'driverTotalPodiums', 'yearsActive',
    'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions',
    'averagePracticePosition', 'lastFPPositionNumber', 'resultsStartingGridPositionNumber', 'numberOfStops',
    'trackRace', 'streetRace', 'turns', 'average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation', 
    'driverDNFCount', 'driverDNFAvg', 'driver_dnf_rate_5_races',
    'recent_dnf_rate_3_races',  'constructor_dnf_rate_3_races', 
    'constructor_dnf_rate_5_races', 'total_experience', 'driverAge',
     ]

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

# 1. Prepare features and target for safety car prediction
def get_features_and_target_safety_car(safety_cars):
    # Example features (customize as needed)
    features = [
        # Track & Weather
    'grandPrixYear', 'grandPrixName',  'circuitId', 
    'grandPrixLaps', 'turns', 'streetRace', 'trackRace',
    'average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation',

    # Practice
    'fp1PositionNumber', 'fp2PositionNumber', 'fp3PositionNumber', 
    'averagePracticePosition', 'lastFPPositionNumber', 'practice_position_std',
    'practice_improvement', 'practice_improvement_x_qual', 'practice_gap_to_teammate',
    'practice_position_improvement_1P_2P', 'practice_position_improvement_2P_3P', 'practice_position_improvement_1P_3P',

    # Qualifying
    'resultsQualificationPositionNumber', 'best_qual_time', 'pole_time_sec',
    'qualifying_gap_to_pole', 'qualifying_position_percentile', 'qual_gap_to_teammate',
    'qualPos_x_avg_practicePos', 'qualPos_x_last_practicePos',

    # Career/Constructor
    'driverTotalRaceStarts', 'constructorTotalRaceStarts', 'yearsActive', 'driverAge', 'driver_age_squared',
    'street_experience', 'track_experience', 'driver_experience', 'constructor_experience',

    # Track Familiarity
    'track_familiarity',

    # Weather Volatility
    'weather_volatility',

    # Combined/Engineered
    'turns_x_weather', 'turns_x_precip', 'turns_x_wind', 'street_x_weather', 'track_x_weather',
    'driver_experience_x_track_familiarity', 'constructor_experience_x_track_familiarity', 
    ]
    target = 'SafetyCarStatus'  # 1 if safety car, 0 if not
    return safety_cars[features], safety_cars[target]

def get_preprocessor_safety_car():
    categorical_features = ['grandPrixYear', 'grandPrixName', 'circuitId',  ]
    numerical_features = [
        # Track & Weather
    
    'grandPrixLaps', 'turns', 'streetRace', 'trackRace',
    'average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation',

    # Practice
    'fp1PositionNumber', 'fp2PositionNumber', 'fp3PositionNumber', 
    'averagePracticePosition', 'lastFPPositionNumber', 'practice_position_std',
    'practice_improvement', 'practice_improvement_x_qual', 'practice_gap_to_teammate',
    'practice_position_improvement_1P_2P', 'practice_position_improvement_2P_3P', 'practice_position_improvement_1P_3P',

    # Qualifying
    'resultsQualificationPositionNumber', 'best_qual_time', 'pole_time_sec',
    'qualifying_gap_to_pole', 'qualifying_position_percentile', 'qual_gap_to_teammate',
    'qualPos_x_avg_practicePos', 'qualPos_x_last_practicePos',

    # Career/Constructor
    'driverTotalRaceStarts', 'constructorTotalRaceStarts', 'yearsActive', 'driverAge', 'driver_age_squared',
    'street_experience', 'track_experience', 'driver_experience', 'constructor_experience',

    # Track Familiarity
    'track_familiarity',

    # Weather Volatility
    'weather_volatility',

    # Combined/Engineered
    'turns_x_weather', 'turns_x_precip', 'turns_x_wind', 'street_x_weather', 'track_x_weather',
    'driver_experience_x_track_familiarity', 'constructor_experience_x_track_familiarity',
    ]
    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', numerical_imputer),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', categorical_imputer),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    return preprocessor

@st.cache_data
def load_safetycars(nrows):
    safety_cars = pd.read_csv(path.join(DATA_DIR, 'f1SafetyCarFeatures.csv'), sep='\t', nrows=nrows)
    safety_cars = safety_cars.drop_duplicates()
    # Drop duplicate rows based on all feature columns
    features, _ = get_features_and_target_safety_car(safety_cars)
    safety_cars = safety_cars.drop_duplicates(subset=features.columns.tolist())
    return safety_cars
safety_cars = load_safetycars(10000)


###### Training model for final racing position prediction

features, _ = get_features_and_target(data)
missing = [col for col in features.columns if col not in data.columns]
if missing:
    st.error(f"The following feature columns are missing from your data: {missing}")
    st.stop()


def train_and_evaluate_model(data):
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    import numpy as np

    X, y = get_features_and_target(data)
    preprocessor = get_preprocessor_position()

    # Check for missing columns in X
    all_preprocessor_columns = []
    for name, _, cols in preprocessor.transformers:
        all_preprocessor_columns.extend(cols)
    missing_cols = [col for col in all_preprocessor_columns if col not in X.columns]
    if missing_cols:
        st.error(f"These columns are missing from your data and required by the preprocessor: {missing_cols}")
        st.stop()

    if y.isnull().any():
        y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess manually
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train_prep, label=y_train)
    dtest = xgb.DMatrix(X_test_prep, label=y_test)

    # Parameters
    params = {
        "objective": "reg:absoluteerror",
        "learning_rate": 0.1,
        "max_depth": 4,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        # "reg_alpha": 0.3,                # L1 regularization
        # "colsample_bytree": 0.80,         # Sample 80% of features per tree
        # "colsample_bylevel": 0.80,        # Sample 80% per tree level
        # "colsample_bynode": 0.80,         # Sample 80% per split
    }

    # Train with early stopping
    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtest, "eval")],
        early_stopping_rounds=15,
        evals_result=evals_result,
        verbose_eval=False,
    )
    
    # Predict
    y_pred = booster.predict(dtest)
    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mean_err = np.mean(y_pred - y_test)

    # Return the expected 5 values
    return booster, mse, r2, mae, mean_err, evals_result


def train_and_evaluate_dnf_model(data):
    from sklearn.linear_model import LogisticRegression
    X, y = get_features_and_target_dnf(data)
    preprocessor = get_preprocessor_dnf()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model


def train_and_evaluate_safetycar_model(data):
    from sklearn.linear_model import LogisticRegression
    X, y = get_features_and_target_safety_car(data)
    if X.isnull().any().any():
        X = X.fillna(X.mean(numeric_only=True))
    y = (y > 0).astype(int)

    model = Pipeline([
        ('preprocessor', get_preprocessor_safety_car()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)

    return model

# @st.cache_resource
def get_trained_model():
    model, mse, r2, mae, mean_err, evals_result = train_and_evaluate_model(data)
    return model

model = get_trained_model()
data['DNF'] = data['DNF'].astype(int)

# Diagnostic: Try Logistic Regression for DNF prediction
from sklearn.linear_model import LogisticRegression

X_dnf, y_dnf = get_features_and_target_dnf(data)
mask = y_dnf.notnull() & np.isfinite(y_dnf)
X_dnf, y_dnf = X_dnf[mask], y_dnf[mask]
preprocessor = get_preprocessor_dnf()
X_dnf_prep = preprocessor.fit_transform(X_dnf)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_dnf_prep, y_dnf)
probs = clf.predict_proba(X_dnf_prep)[:, 1]


dnf_model = train_and_evaluate_dnf_model(data)

safetycar_model = train_and_evaluate_safetycar_model(safety_cars)


X_sc, y_sc = get_features_and_target_safety_car(safety_cars)
if X_sc.isnull().any().any():
    
    X_sc = X_sc.fillna(X_sc.mean(numeric_only=True))


def monte_carlo_feature_selection(X, y, model_class, n_trials=50, min_features=8, max_features=15, random_state=42):
    import random
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    results = []
    feature_names = X.columns.tolist()
    rng = random.Random(random_state)
    for i in range(n_trials):
        subset = rng.sample(feature_names, k=rng.randint(min_features, max_features))
        X_subset = X[subset].copy()
        # Convert object columns to category codes for XGBoost
        for col in X_subset.select_dtypes(include='object').columns:
            X_subset[col] = X_subset[col].astype('category').cat.codes
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=i)
        # --- Clean y_train and y_test ---
        mask_train = y_train.notnull() & np.isfinite(y_train)
        mask_test = y_test.notnull() & np.isfinite(y_test)
        X_train, y_train = X_train[mask_train], y_train[mask_train]
        X_test, y_test = X_test[mask_test], y_test[mask_test]
        # Skip if not enough data
        if len(y_train) == 0 or len(y_test) == 0:
            continue
        model = model_class()
        model.fit(X_train, y_train)

        preprocessor = get_preprocessor_position()
        preprocessor.fit(X_train)  # Fit on training data
        X_test_prep = preprocessor.transform(X_test)

        y_pred = model.predict(xgb.DMatrix(X_test_prep))
        
        mae = mean_absolute_error(y_test, y_pred)
        results.append({'features': subset, 'mae': mae})
    return results

def run_rfe_feature_selection(X, y, n_features_to_select=10):
    """Run Recursive Feature Elimination (RFE) with XGBoost."""
    estimator = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=42)
    rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_].tolist()
    ranking = rfe.ranking_
    return selected_features, ranking

def run_boruta_feature_selection(X, y, max_iter=200):
    """Run Boruta feature selection with XGBoost."""
    X_boruta = X.copy()
    for col in X_boruta.select_dtypes(include='object').columns:
        X_boruta[col] = X_boruta[col].astype('category').cat.codes
    for col in X_boruta.select_dtypes(include='Int64').columns:
        X_boruta[col] = X_boruta[col].astype(float)
    X_boruta = X_boruta.replace({pd.NA: np.nan})
    y_boruta = y.replace({pd.NA: np.nan})
    X_boruta = X_boruta.fillna(X_boruta.mean(numeric_only=True))
    y_boruta = y_boruta.fillna(y_boruta.mean())
    # Remove columns that are all NaN
    all_nan_cols = X_boruta.columns[X_boruta.isnull().all()]
    if len(all_nan_cols) > 0:
        st.warning(f"Removing columns with all NaN values: {list(all_nan_cols)}")
        X_boruta = X_boruta.drop(columns=all_nan_cols)
    # Drop rows with any remaining NaN
    mask = (~X_boruta.isnull().any(axis=1)) & (~y_boruta.isnull())
    X_boruta = X_boruta[mask]
    y_boruta = y_boruta[mask]
    if len(X_boruta) == 0 or len(y_boruta) == 0:
        st.warning("No data available after cleaning for Boruta feature selection. Please check your data or filters.")
        return [], []
    for col in X_boruta.columns:
        if X_boruta[col].dtype not in [np.float64, np.int64]:
            X_boruta[col] = X_boruta[col].astype(float)
    # --- Ensure y_boruta is a 1D numpy array ---
    if isinstance(y_boruta, pd.DataFrame):
        y_boruta = y_boruta.iloc[:, 0]
    y_boruta = np.asarray(y_boruta).ravel()
    estimator = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=42)
    boruta_selector = BorutaPy(estimator, n_estimators='auto', verbose=0, random_state=42, max_iter=max_iter)
    boruta_selector.fit(X_boruta.values, y_boruta)
    selected_features = X_boruta.columns[boruta_selector.support_].tolist()
    ranking = boruta_selector.ranking_
    return selected_features, ranking

def rfe_minimize_mae(X, y, min_features=3, max_features=20, step=1, random_state=42):
    """Run RFE for a range of feature counts and return the subset with the lowest MAE."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    # Convert object columns to category codes
    X_rfe = X.copy()
    for col in X_rfe.select_dtypes(include='object').columns:
        X_rfe[col] = X_rfe[col].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, random_state=random_state)
    best_mae = float('inf')
    best_features = None
    best_ranking = None
    maes = []
    for n_features in range(min_features, min(max_features, X_rfe.shape[1]) + 1, step):
        estimator = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=random_state)
        rfe = RFE(estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X_train, y_train)
        selected = X_rfe.columns[rfe.support_].tolist()
        # Train model on selected features
        model = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=random_state)
        model.fit(X_train[selected], y_train)
        y_pred = model.predict(X_test[selected])
        mae = mean_absolute_error(y_test, y_pred)
        maes.append((n_features, mae))
        if mae < best_mae:
            best_mae = mae
            best_features = selected
            best_ranking = rfe.ranking_
    return best_features, best_ranking, best_mae, maes

if st.checkbox('Filter Results'):
    # Create a dictionary to store selected filters for multiple columns
    filters = {}
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
            
            filters[column] = selected_range

        elif data[column].dtype ==bool:
            
           
            selected_value = st.sidebar.checkbox(
                column_friendly_name,
                value=False,
                key=f"checkbox_filter_{column}"
            )
            if selected_value:
                filters[column] = True

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

    st.write(f"Number of filtered results: {len(filtered_data):,d}")
    filtered_data = filtered_data.sort_values(by=['grandPrixYear', 'resultsFinalPositionNumber'], ascending=[False, True])
    filtered_data = filtered_data.drop_duplicates()
    st.dataframe(filtered_data, column_config=columns_to_display, column_order=['grandPrixYear', 'grandPrixName', 'streetRace', 'trackRace', 'constructorName', 'resultsDriverName', 'resultsPodium', 'resultsTop5',
         'resultsTop10','resultsStartingGridPositionNumber','resultsFinalPositionNumber','positionsGained', 'DNF', 'resultsQualificationPositionNumber',
           'q1End', 'q2End', 'q3Top10', 'averagePracticePosition',  'lastFPPositionNumber','numberOfStops', 'averageStopTime', 'totalStopTime',
           'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalPolePositions', 'resultsReasonRetired',
           'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 'positionsGained', 'avgLapTime', 'finishingTime'
           ], hide_index=True, width=2400, height=600)

    positionCorrelation = filtered_data[[
    'lastFPPositionNumber', 'resultsFinalPositionNumber', 'resultsStartingGridPositionNumber','grandPrixLaps', 'averagePracticePosition', 'DNF', 'resultsTop10', 'resultsTop5', 'resultsPodium', 'streetRace', 'trackRace',
    'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions', 'turns', 'positionsGained', 'q1End', 'q2End', 'q3Top10',  'driverBestStartingGridPosition', 'yearsActive',
    'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalPolePositions', 'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 'positionsGained',
    'avgLapPace', 'finishingTime']].corr(method='pearson')

    ## Rename Correlation Rows
    positionCorrelation.index=['Last FP.', 'Final Pos.' ,'Starting Grid Pos.', 'Laps', 'Avg Practice Pos.', 
     'DNF', 'Top 10', 'Top 5', 'Podium', 'Street', 'Track', 'Constructor Race Starts', 'Constructor Total Race Wins', 'Constructor Pole Pos.',
     'Turns', 'Positions Gained', 'Out at Q1', 'Out at Q2', 'Q3 Top 10', 'Best Starting Grid Pos.', 'Years Active',
     'Best Result', 'Total Championship Wins', 'Total Pole Positions', 'Race Entries', 'Race Starts', 'Race Wins',
    'Race Laps', 'Total Podiums', 'Positions Gained', 'Avg. Lap Pace', 'Finishing Time']

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
        avg_practice_position_vs_final_position_slope = slope
        avg_practice_position_vs_final_position_intercept = intercept
        # Display regression statistics
        st.write(f"**Regression Statistics:**")
        st.write(f"R-squared: {r_value**2:.2f}")
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

    else:
        st.write("Not enough data for regression analysis.")

    # Ensure unique index and columns
    positionCorrelation = positionCorrelation.loc[~positionCorrelation.index.duplicated(keep='first')]
    positionCorrelation = positionCorrelation.loc[:, ~positionCorrelation.columns.duplicated(keep='first')]

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

    # Count the number of entries (drivers) for each driver
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
        .groupby(['grandPrixName'])
        .size()
        .reset_index(name='race_entry_count')
)

    race_dnf_counts = (
    filtered_data[filtered_data['DNF']]
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


    constructor_dnf_counts = (
    filtered_data[filtered_data['DNF']]
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

    # Extract features and target
    X, y = get_features_and_target(filtered_data)

    if len(X) == 0 or len(y) == 0:
        st.warning("No data available after filtering. Please adjust your filters.")
    else:
        # Split the data
        model, mse, r2, mae, mean_err, evals_result = train_and_evaluate_model(filtered_data)

        st.write(f"Mean Squared Error: {mse:.3f}")

        st.write(f"R^2 Score: {r2:.3f}")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"Mean Error: {mean_err:.2f}")
               
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        preprocessor = get_preprocessor_position()
        preprocessor.fit(X_train)  # Fit on training data
        X_test_prep = preprocessor.transform(X_test)

        y_pred = model.predict(xgb.DMatrix(X_test_prep))

        # Create a DataFrame to display the features and predictions
        results_df = X_test.copy()
        results_df['Actual'] = y_test.values
        results_df['Predicted'] = y_pred
        results_df['Error'] = results_df['Actual'] - results_df['Predicted']

        # Select the top 3 actual finishers in each race (or just overall if not grouped by race)
        top3_actual = results_df.nsmallest(3, 'Actual')

        # Calculate MAE for the top 3
        mae_top3 = mean_absolute_error(top3_actual['Actual'], top3_actual['Predicted'])
        st.write(f"Mean Absolute Error (MAE) for Top 3 Podium Drivers: {mae_top3:.3f}")

        # Optionally, display the top 3 actual vs predicted
        st.subheader("Top 3 Podium Drivers: Actual vs Predicted")
        st.dataframe(top3_actual[['grandPrixName', 'constructorName', 'resultsDriverName', 'Actual', 'Predicted', 'Error']], hide_index=True)

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

        # Display feature importances
        st.subheader("Feature Importance")
        
        # Get feature names from your preprocessor
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

        # Get importances from Booster
        importances_dict = model.get_score(importance_type='weight')

        # Map Booster's feature names (e.g., 'f0', 'f1', ...) to your actual feature names
        importances = []
        for i, name in enumerate(feature_names):
            importances.append(importances_dict.get(f'f{i}', 0))

        # Create DataFrame for display
        feature_importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Percentage': np.array(importances) / (np.sum(importances) or 1) * 100
        }).sort_values(by='Importance', ascending=False)

        st.dataframe(feature_importances_df, hide_index=True, width=800)

        # Clean up feature names by removing 'num__'
        feature_names = [name.replace('num__', '') for name in feature_names]
        feature_names = [name.replace('cat__', '') for name in feature_names]

        # Create a DataFrame for feature importances
        feature_importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Percentage': importances / np.sum(importances) * 100,

        }).sort_values(by='Importance', ascending=False)

        # Display the top 50 features
        st.dataframe(feature_importances_df.head(50), hide_index=True, width=800)
        # Display the predictive data model without index
        # -----------------------------


if st.checkbox(f"Show {current_year} Schedule"):
    st.title(f"{current_year} Races:")
    
    raceSchedule = raceSchedule[raceSchedule['year'] == current_year]
    
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
    
    last_race = detailsOfNextRace.iloc[1]

    active_driver_ids = data[data['activeDriver'] == True]['resultsDriverId'].unique()
    all_drivers_df = pd.DataFrame({'resultsDriverId': active_driver_ids})

    # Merge with detailsOfNextRace to get historical data if available
    input_data_next_race = all_drivers_df.merge(
        detailsOfNextRace,
        on='resultsDriverId',
        how='left'
    )
    
    # Convert all Int64 columns to Float64 before filling NaN with mean
    for col in input_data_next_race.select_dtypes(include='Int64').columns:
        input_data_next_race[col] = input_data_next_race[col].astype('Float64')
    # Fill missing values with mean (or other default)
    input_data_next_race = input_data_next_race.fillna(input_data_next_race.mean(numeric_only=True))

    input_data_next_race = input_data_next_race.drop(columns=['firstName', 'lastName'], errors='ignore')
    # Fill missing driver and constructor names from reference tables
    input_data_next_race = pd.merge(
    input_data_next_race,
    drivers[['id', 'firstName', 'lastName']],
    left_on='resultsDriverId',
    right_on='id',
    how='left'
    )
    input_data_next_race['resultsDriverName'] = input_data_next_race['resultsDriverName'].fillna(
        input_data_next_race['firstName'].fillna('') + ' ' + input_data_next_race['lastName'].fillna('')
    )

    # Fill missing constructor names from reference data
    constructor_ref = data[['resultsDriverId', 'constructorName']].drop_duplicates(subset=['resultsDriverId', 'constructorName'])
    input_data_next_race = pd.merge(
        input_data_next_race,
        constructor_ref,
        on='resultsDriverId',
        how='left',
        suffixes=('', '_ref')
    )

    input_data_next_race['constructorName'] = input_data_next_race['constructorName'].fillna(input_data_next_race['constructorName_ref'])
    input_data_next_race = input_data_next_race.drop(columns=['constructorName_ref'], errors='ignore')


    features, _ = get_features_and_target(data)
    feature_names = features.columns.tolist()

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

    all_active_driver_inputs = input_data_next_race[feature_names + ['resultsDriverId', 'Abbreviation']]

    # Get latest stats for each active driver
    latest_stats = (
        data.sort_values('grandPrixYear')
            .groupby('resultsDriverId')
            .tail(1)[['resultsDriverId', 'yearsActive', 'driverTotalRaceStarts']]
    )

    all_active_driver_inputs = pd.merge(
        all_active_driver_inputs,
        latest_stats,
        left_on='resultsDriverId',
        right_on='resultsDriverId',
        how='left',
        suffixes=('', '_latest')
    )


    for col in ['yearsActive', 'driverTotalRaceStarts']:
        latest_col = f"{col}_latest"
        if latest_col in all_active_driver_inputs.columns:
            # Only combine if the column is not empty or all-NA
            if not all_active_driver_inputs[latest_col].isnull().all():
                all_active_driver_inputs[col] = all_active_driver_inputs[latest_col].combine_first(all_active_driver_inputs[col])
            all_active_driver_inputs = all_active_driver_inputs.drop(columns=[latest_col], errors='ignore')
 
    
    # all_active_driver_inputs = input_data_next_race[feature_names + ['resultsDriverId']]
    all_active_driver_inputs = pd.merge(all_active_driver_inputs, practices, left_on='resultsDriverId', right_on='resultsDriverId', how='left')
    if 'resultsDriverId_x' in all_active_driver_inputs.columns:
        # If resultsDriverId_x exists, it means there was a merge conflict
        all_active_driver_inputs = all_active_driver_inputs.rename(columns={'resultsDriverId_x': 'resultsDriverId'})
    elif 'resultsDriverId_y' in all_active_driver_inputs.columns:
        # If resultsDriverId_y exists, it means there was a merge conflict
        all_active_driver_inputs = all_active_driver_inputs.rename(columns={'resultsDriverId_y': 'resultsDriverId'})
    
    all_active_driver_inputs = all_active_driver_inputs.drop_duplicates(subset=['resultsDriverId'])
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
    # st.write(all_active_driver_inputs.columns.tolist())
    all_active_driver_inputs = pd.merge(
        all_active_driver_inputs, 
        qualifying, 
        left_on='Abbreviation', 
        right_on='Abbreviation', 
        how='left',
        suffixes=('_datamodel', '_qualifying')
    )
    
    if 'best_qual_time_qualifying' in all_active_driver_inputs.columns:
        all_active_driver_inputs.rename(columns={'best_qual_time_qualifying': 'best_qual_time'}, inplace=True)

    if 'teammate_qual_delta_qualifying' in all_active_driver_inputs.columns:
        all_active_driver_inputs.rename(columns={'teammate_qual_delta_qualifying': 'teammate_qual_delta'}, inplace=True)

    if 'teammate_practice_delta_x' in all_active_driver_inputs.columns:
        all_active_driver_inputs.rename(columns={'teammate_practice_delta_x': 'teammate_practice_delta'}, inplace=True)

    if 'BestConstructorPracticeLap_sec' not in all_active_driver_inputs.columns:
        if 'BestConstructorPracticeLap_sec_x' in all_active_driver_inputs.columns:
            all_active_driver_inputs.rename(columns={'BestConstructorPracticeLap_sec_x': 'BestConstructorPracticeLap_sec'}, inplace=True)

    all_active_driver_inputs = all_active_driver_inputs.rename(columns={'Points_datamodel': 'Points', 'totalChampionshipPoints_datamodel': 'totalChampionshipPoints',
        'totalPolePositions_datamodel': 'totalPolePositions','totalFastestLaps_datamodel': 'totalFastestLaps'} )    

    # Remove duplicate columns (if any)
    all_active_driver_inputs = all_active_driver_inputs.loc[:, ~all_active_driver_inputs.columns.duplicated()]


    existing_feature_names = [col for col in feature_names if col in all_active_driver_inputs.columns]

    X_predict = all_active_driver_inputs[existing_feature_names]
    
    # commented out on 9/17/2025 for early stopping
    # predicted_position = model.predict(X_predict)
    preprocessor = get_preprocessor_position()
    missing_cols = [col for col in preprocessor.transformers[0][2] + preprocessor.transformers[1][2] if col not in X_predict.columns]
    if missing_cols:
        st.error(f"These columns are missing from your prediction data and required by the preprocessor: {missing_cols}")
        st.stop()

    preprocessor.fit(X_predict)  # Fit if not already fitted, or reuse fitted preprocessor
    X_predict_prep = preprocessor.transform(X_predict)
    predicted_position = model.predict(xgb.DMatrix(X_predict_prep))

    # Get DNF feature names
    dnf_features, _ = get_features_and_target_dnf(data)
    dnf_feature_names = dnf_features.columns.tolist()

    # For position prediction
    for col in feature_names:
        if col not in all_active_driver_inputs.columns:
            all_active_driver_inputs[col] = np.nan

    # For DNF prediction
    for col in dnf_feature_names:
        if col not in all_active_driver_inputs.columns:
            all_active_driver_inputs[col] = np.nan

    existing_dnf_features = [col for col in dnf_feature_names if col in all_active_driver_inputs.columns]
    missing_dnf_features = [col for col in dnf_feature_names if col not in all_active_driver_inputs.columns]
    if missing_dnf_features:
        st.warning(f"These DNF features are missing from prediction data and will be skipped: {missing_dnf_features}")

    X_predict_dnf = all_active_driver_inputs[existing_dnf_features]

    if X_predict_dnf.isnull().any().any():
        # st.warning("Imputing missing values in X_predict_dnf before prediction.")
        X_predict_dnf = X_predict_dnf.fillna(X_predict_dnf.mean(numeric_only=True))
    predicted_dnf_proba = dnf_model.predict_proba(X_predict_dnf)[:, 1]  # Probability of DNF=True

    
    # Holdout year evaluation for Safety Car Model
    train = safety_cars[safety_cars['grandPrixYear'] < current_year]
    test = safety_cars[safety_cars['grandPrixYear'] == current_year]
    X_train, y_train = get_features_and_target_safety_car(train)
    X_test, y_test = get_features_and_target_safety_car(test)

    holdout_model = train_and_evaluate_safetycar_model(train)
    # Now use holdout_model for predictions:
    # y_pred = holdout_model.predict_proba(X_test)[:, 1]
    if X_test.isnull().any().any():
        
        X_test = X_test.fillna(X_test.mean(numeric_only=True))
    y_pred = holdout_model.predict_proba(X_test)[:, 1]
   
    # Find the most recent year with both classes present in test set
    holdout_year = None
    for year in sorted(safety_cars['grandPrixYear'].unique(), reverse=True):
        test = safety_cars[safety_cars['grandPrixYear'] == year]
        if len(test['SafetyCarStatus'].unique()) > 1:
            holdout_year = year
            break

    if holdout_year is not None:
        train = safety_cars[safety_cars['grandPrixYear'] < holdout_year]
        test = safety_cars[safety_cars['grandPrixYear'] == holdout_year]
        X_train, y_train = get_features_and_target_safety_car(train)
        X_test, y_test = get_features_and_target_safety_car(test)

        # Do NOT re-fit safetycar_model! Instead, create a new model for holdout/test:
        holdout_model = train_and_evaluate_safetycar_model(train)
        y_pred = holdout_model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score
        # st.write(f"Safety Car ROC AUC (holdout year {holdout_year}):", roc_auc_score(y_test, y_pred))
    else:
        st.write("No valid holdout year with both classes present.")

    # Get race-level features for the next race (should be one row)
    race_level = detailsOfNextRace.drop_duplicates(subset=['grandPrixYear', 'grandPrixName'])

    features, _ = get_features_and_target_safety_car(safety_cars)
    safetycar_feature_columns = features.columns.tolist()


    synthetic_row = {col: np.nan for col in safetycar_feature_columns}

    synthetic_row['grandPrixYear'] = nextRace['year'].values[0]
    synthetic_row['grandPrixName'] = nextRace['fullName'].values[0]
    
    # Fill with available info from nextRace, schedule, weather, etc.
    for col in ['circuitId', 'grandPrixLaps', 'turns', 'streetRace', 'trackRace']:
        if col in nextRace.columns:
            synthetic_row[col] = nextRace[col].values[0]

    # Fill weather features if available
    weather_row = weatherData[weatherData['grandPrixId'] == next_race_id]
    if not weather_row.empty:
        for col in ['average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation']:
            if col in weather_row.columns:
                synthetic_row[col] = weather_row[col].values[0]

    
    safety_cars['SafetyCarStatus'] = (safety_cars['SafetyCarStatus'] > 0).astype(int)

    # Improved logic: Use median of last 2 years for this GP if available, else overall median
    for col in safetycar_feature_columns:
        if pd.isna(synthetic_row[col]) and col in safety_cars.columns:
            if pd.api.types.is_numeric_dtype(safety_cars[col]):
                gp_vals = safety_cars[safety_cars['grandPrixName'] == synthetic_row['grandPrixName']]
                # Aggregate by year (mean across drivers for each year)
                per_race_means = gp_vals.groupby('grandPrixYear')[col].mean()
                # Get the last 2 years
                last_2_years = sorted(per_race_means.index)[-2:]
                per_race_means_recent = per_race_means.loc[last_2_years]
                if not per_race_means_recent.empty:
                    synthetic_row[col] = per_race_means_recent.median()
                else:
                    synthetic_row[col] = safety_cars[col].dropna().median()
            else:
                synthetic_row[col] = np.nan

    synthetic_df = pd.DataFrame([synthetic_row])


    features, _ = get_features_and_target_safety_car(safety_cars)
    feature_list = features.columns.tolist()

    X_predict_safetycar = synthetic_df[feature_list]
    if X_predict_safetycar.isnull().any().any():

        X_predict_safetycar = X_predict_safetycar.fillna(X_predict_safetycar.mean(numeric_only=True))
    safety_car_proba = safetycar_model.predict_proba(X_predict_safetycar)[:, 1][0]

    # Add both to your DataFrame
    all_active_driver_inputs['PredictedFinalPosition'] = predicted_position
    all_active_driver_inputs['PredictedDNFProbability'] = predicted_dnf_proba
    all_active_driver_inputs['PredictedDNFProbabilityPercentage'] = (all_active_driver_inputs['PredictedDNFProbability'] * 100).round(3)
    all_active_driver_inputs['driverDNFCount'] = all_active_driver_inputs['driverDNFCount'].fillna(0).astype(int)
    all_active_driver_inputs['driverDNFAvg'] = all_active_driver_inputs['driverDNFAvg'].fillna(0).astype(float)
    all_active_driver_inputs['driverDNFPercentage'] = (all_active_driver_inputs['driverDNFAvg'].fillna(0).astype(float) * 100).round(3)

    # --- Rookie DNF Simulation: Overwrite rookie DNF predictions with simulation ---
    all_active_driver_inputs = simulate_rookie_dnf(data, all_active_driver_inputs, current_year, n_simulations=1000)
    if 'PredictedDNFProbabilityStd' not in all_active_driver_inputs.columns:
        all_active_driver_inputs['PredictedDNFProbabilityStd'] = np.nan

    # --- Rookie Simulation: Overwrite rookie predictions with simulation ---
    all_active_driver_inputs = simulate_rookie_predictions(data, all_active_driver_inputs, current_year, n_simulations=1000)
    if 'PredictedFinalPositionStd' not in all_active_driver_inputs.columns:
        all_active_driver_inputs['PredictedFinalPositionStd'] = np.nan

    all_active_driver_inputs.sort_values(by='PredictedFinalPosition', ascending=True, inplace=True)
    all_active_driver_inputs['Rank'] = range(1, len(all_active_driver_inputs) + 1)
    all_active_driver_inputs = all_active_driver_inputs.set_index('Rank')

    # Fix the column data for display after the merge
    all_active_driver_inputs.drop(columns=['constructorName', 'constructorName_y'], inplace=True, errors='ignore')
    all_active_driver_inputs = all_active_driver_inputs.rename(columns={'constructorName_x': 'constructorName'})

    # st.write(all_active_driver_inputs.columns.tolist())
    st.subheader("Predictive Results for Active Drivers")

    st.dataframe(all_active_driver_inputs, hide_index=False, column_config=predicted_position_columns_to_display, width=800, height=800, 
    column_order=['constructorName', 'resultsDriverName', 'PredictedFinalPosition', 'PredictedFinalPositionStd'])    

    st.subheader("Predictive DNF")

    st.write("Logistic Regression DNF Probabilities:")
    st.write("Min:", probs.min(), "Max:", probs.max(), "Mean:", probs.mean())

    all_active_driver_inputs.sort_values(by='PredictedDNFProbabilityPercentage', ascending=False, inplace=True)
    st.dataframe(all_active_driver_inputs, hide_index=False, column_config=predicted_dnf_position_columns_to_display, width=800, height=800, 
    column_order=['constructorName', 'resultsDriverName', 'driverDNFCount',  'driverDNFPercentage', 'PredictedDNFProbabilityPercentage', 'PredictedDNFProbabilityStd'], )  

    st.subheader("Predicted Safety Car")

    # Ensure race_level is a copy to avoid SettingWithCopyWarning
    race_level = race_level.copy()  # Add this line before assignment if not already a copy
    
    X_sc, y_sc = get_features_and_target_safety_car(safety_cars)
    if X_sc.isnull().any().any():

        X_sc = X_sc.fillna(X_sc.mean(numeric_only=True))
    safety_cars['PredictedSafetyCarProbability'] = safetycar_model.predict_proba(X_sc)[:, 1]
    safety_cars['PredictedSafetyCarProbabilityPercentage'] = (safety_cars['PredictedSafetyCarProbability'] * 100).round(3)

    historical_display = safety_cars[['grandPrixName', 'grandPrixYear', 'PredictedSafetyCarProbabilityPercentage']].copy()
    historical_display['Type'] = 'Historical'

    synthetic_df['PredictedSafetyCarProbability'] = safety_car_proba
    synthetic_df['PredictedSafetyCarProbabilityPercentage'] = (synthetic_df['PredictedSafetyCarProbability'] * 100).round(3)
    synthetic_df['Type'] = 'Next Race'

    # Only show historical and synthetic predictions for the current Grand Prix
    current_gp_name = synthetic_df['grandPrixName'].values[0]
    current_gp_year = synthetic_df['grandPrixYear'].values[0]

    # Filter and deduplicate historical predictions for this Grand Prix
    historical_this_gp = historical_display[historical_display['grandPrixName'] == current_gp_name].copy()
    historical_this_gp = historical_this_gp[historical_this_gp['grandPrixYear'] != current_gp_year]
    historical_this_gp = historical_this_gp.drop_duplicates(subset=['grandPrixYear'])

    # Combine historical and synthetic predictions
    display_df = pd.concat([
        historical_this_gp,
        synthetic_df[['grandPrixName', 'grandPrixYear', 'PredictedSafetyCarProbabilityPercentage', 'Type']]
    ], ignore_index=True)

    st.write("Historical Safety Car Probabilities (mean):", safety_cars['PredictedSafetyCarProbabilityPercentage'].mean())
    st.write("Historical Safety Car Probabilities (min/max):", safety_cars['PredictedSafetyCarProbabilityPercentage'].min(), safety_cars['PredictedSafetyCarProbabilityPercentage'].max())

    st.dataframe(
        display_df[['grandPrixName', 'grandPrixYear', 'PredictedSafetyCarProbabilityPercentage', 'Type']].sort_values(by=['grandPrixYear'], ascending=[False]),
        hide_index=True,
        width=800,
        height=400,
        column_config={
            'grandPrixName': st.column_config.TextColumn("Grand Prix"),
            'grandPrixYear': st.column_config.NumberColumn("Year"),
            'PredictedSafetyCarProbabilityPercentage': st.column_config.NumberColumn("Predicted Safety Car Probability (%)"),
            'Type': st.column_config.TextColumn("Type")
    }
)

    # After prediction and before displaying the predictive results
    predicted_results = all_active_driver_inputs.reset_index()[[
        'Rank', 'resultsDriverName', 'constructorName', 'PredictedFinalPosition', 'PredictedDNFProbability', 'PredictedDNFProbabilityPercentage'
    ]].copy()
    predicted_results['raceId'] = next_race_id
    predicted_results['grandPrixName'] = nextRace['fullName'].values[0]
    predicted_results['grandPrixYear'] = nextRace['year'].values[0]

    # Save to CSV for later comparison
    predicted_results.to_csv(path.join(DATA_DIR, f"predictions_{next_race_id}_{nextRace['year'].values[0]}.csv"), index=False)

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

    st.subheader(f"Flags and Safety Cars from {nextRace['fullName'].head(1).values[0]}:")
    st.caption("Race messages, including flags, are only available going back to 2018.")
    # race_control_messages_grouped_with_dnf.csv
    raceMessagesOfNextRace = race_messages[race_messages['grandPrixId'] == next_race_id]
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

show_advanced = st.checkbox("Advanced Options:")

if show_advanced:
    
    model, mse, r2, mae, mean_err, evals_result = train_and_evaluate_model(data)

    if st.checkbox('Show Predictive Data Model'):
        st.subheader("Predictive Data Model")
        
        
        
        model, mse, r2, mae, mean_err, evals_result = train_and_evaluate_model(data)

        st.write(f"Mean Squared Error: {mse:.3f}")
        st.write(f"R^2 Score: {r2:.3f}")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"Mean Error: {mean_err:.2f}")

        # Show boosting rounds used (only here, not at the top)
        st.write(f"Boosting rounds used: {model.best_iteration + 1}")

        # Extract features and target
        X, y = get_features_and_target(data)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        preprocessor = get_preprocessor_position()
        preprocessor.fit(X_train)  # Fit on training data
        X_test_prep = preprocessor.transform(X_test)

        y_pred = model.predict(xgb.DMatrix(X_test_prep))

        # Create a DataFrame to display the features and predictions
        results_df = X_test.copy()
        results_df['Actual'] = y_test.values
        results_df['Predicted'] = y_pred

        # Assuming results_df has columns: 'resultsDriverName', 'Actual', 'Predicted'
        results_df['Error'] = results_df['Actual'] - results_df['Predicted']
        results_df['AbsError'] = results_df['Error'].abs()
        results_df['SquaredError'] = results_df['Error'] ** 2

        # Group by driver and calculate ME and MAE
        driver_error_stats = results_df.groupby('resultsDriverName').agg(
            MeanError=('Error', 'mean'),
            MeanAbsoluteError=('AbsError', 'mean'),
            RMSE=('SquaredError', lambda x: np.sqrt(np.mean(x))),
            MedianAbsoluteError=('AbsError', 'median'),
            MaxError=('Error', 'max'),
            MinError=('Error', 'min'),
            Count=('Error', 'count')
        ).reset_index()

        # Display in Streamlit
        st.subheader("Mean Error (ME) and Mean Absolute Error (MAE) per Driver")
        st.write(f"Total number of drivers: {len(driver_error_stats)}")
        st.write(f"Total number of results: {len(results_df)}")
        driver_error_stats = driver_error_stats.sort_values(by='MeanAbsoluteError', ascending=False)
        driver_error_stats['MeanError'] = driver_error_stats['MeanError'].round(3)
        driver_error_stats['MeanAbsoluteError'] = driver_error_stats['MeanAbsoluteError'].round(3)
        driver_error_stats['RMSE'] = driver_error_stats['RMSE'].round(3)
        driver_error_stats['MedianAbsoluteError'] = driver_error_stats['MedianAbsoluteError'].round(3)
        driver_error_stats['MaxError'] = driver_error_stats['MaxError'].round(3)
        driver_error_stats['MinError'] = driver_error_stats['MinError'].round(3)
        driver_error_stats['Count'] = driver_error_stats['Count'].astype(int)
        driver_error_stats = driver_error_stats.rename(columns={
            'resultsDriverName': 'Driver',
            'MeanError': 'Mean Error',
            'MeanAbsoluteError': 'Mean Absolute Error',
            'RMSE': 'Root Mean Squared Error',
            'MedianAbsoluteError': 'Median Absolute Error',
            'MaxError': 'Max Error',
            'MinError': 'Min Error',
            'Count': 'Number of Results'
        })
        st.subheader("Error Metrics per Driver")
        st.dataframe(driver_error_stats, hide_index=True, width=1000)


        # Display the first 15 rows
        st.subheader("Predictive Results with Features")
        st.dataframe(results_df, hide_index=True, width=800)

        # Display feature importances
        st.subheader("Feature Importances")

        feature_names = get_features_and_target(data)[0].columns.tolist()
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

        # Get importances from Booster
        importances_dict = model.get_score(importance_type='weight')

        # Map Booster's feature names (e.g., 'f0', 'f1', ...) to your actual feature names
        importances = []
        for i, name in enumerate(feature_names):
            importances.append(importances_dict.get(f'f{i}', 0))

        feature_importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Percentage': np.array(importances) / (np.sum(importances) or 1) * 100
        }).sort_values(by='Importance', ascending=False)

        # Show boosting rounds used (only here, not at the top)
        st.write(f"Boosting rounds used: {model.best_iteration + 1}")

        st.dataframe(feature_importances_df, hide_index=True, width=800)

    if st.checkbox("Show Permutation Importance (Least Helpful Features)"):
        st.subheader("Permutation Importance (Feature Impact on Model Error)")
        from sklearn.inspection import permutation_importance

        # Get features and target
        X, y = get_features_and_target(data)
        mask = y.notnull() & np.isfinite(y)
        X, y = X[mask], y[mask]
        preprocessor = get_preprocessor_position()
        X_prep = preprocessor.fit_transform(X)

        # Fit model
        model = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=42)
        model.fit(X_prep, y)

        # Run permutation importance
        result = permutation_importance(model, X_prep, y, n_repeats=10, random_state=42)
        importances = result.importances_mean
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

        perm_df = pd.DataFrame({
            'Feature': feature_names,
            'Permutation Importance': importances
        }).sort_values(by='Permutation Importance', ascending=True)

        st.write("Features with lowest permutation importance (least helpful):")
        st.dataframe(perm_df.head(50), hide_index=True, width=800)
        st.write("Features with highest permutation importance (most helpful):")
        st.dataframe(perm_df.tail(50).sort_values(by='Permutation Importance', ascending=False), hide_index=True, width=800)

    if st.checkbox("Show High-Cardinality Features (Overfitting Risk)"):
        st.subheader("High-Cardinality Features (Potential Overfitting Risk)")
        X, _ = get_features_and_target(data)
        cardinality = X.nunique().sort_values(ascending=False)
        cardinality_df = pd.DataFrame({
            'Feature': cardinality.index,
            'Unique Values': cardinality.values
        })
        # Highlight features with >50 unique values (you can adjust this threshold)
        cardinality_df['Risk'] = np.where(cardinality_df['Unique Values'] > 50, 'High', 'Low')
        st.write("Features with high cardinality (many unique values) are more likely to cause overfitting, especially if they are IDs or post-event info.")
        st.dataframe(cardinality_df, hide_index=True, width=800)

    if st.checkbox("Early Stopping Details"):
        st.subheader("Early Stopping & Most Important Feature")

        # 1. Where early stopping occurred
        # mae_per_round = evals_result['eval']['absolute_error'] if 'absolute_error' in evals_result['eval'] else evals_result['eval']['mae']
        # # best_round = model.best_iteration
        
        
        # if hasattr(model, "best_iteration"):
        #     best_round = model.best_iteration
        #     lowest_mae = mae_per_round[best_round]
        #     st.write(f"Early stopping occurred at round {best_round + 1} (lowest MAE: {lowest_mae:.4f})")
        # else:
        #     # For Booster object, use num_boosted_rounds if available
        #     if hasattr(model, "num_boosted_rounds"):
        #         best_round = model.num_boosted_rounds()
        #         lowest_mae = mae_per_round[best_round]
        #         st.write(f"Early stopping was not used. Model ran for {best_round} boosting rounds.")
        #     else:
        #         st.write("Early stopping was not used or best_iteration is not available.")
        # # st.write(f"Early stopping occurred at round {best_round + 1} (lowest MAE: {lowest_mae:.4f})")
        # st.line_chart(mae_per_round)
        mae_per_round = evals_result['eval']['absolute_error'] if 'absolute_error' in evals_result['eval'] else evals_result['eval']['mae']
        best_round = int(np.argmin(mae_per_round))  # Index of lowest MAE
        lowest_mae = mae_per_round[best_round]
        st.write(f"Early stopping occurred at round {best_round + 1} (lowest MAE: {lowest_mae:.4f})")
        st.line_chart(mae_per_round)


        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

        # --- FIX: Define importances here ---
        # importances_dict = model.get_score(importance_type='weight')
        importances_dict = model.get_booster().get_score(importance_type='weight')
        importances = []
        for i, name in enumerate(feature_names):
            importances.append(importances_dict.get(f'f{i}', 0))

        # 2. Most important feature after training
        feature_importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Percentage': np.array(importances) / (np.sum(importances) or 1) * 100
        }).sort_values(by='Importance', ascending=False)

        top_feature = feature_importances_df.iloc[0]
        st.write(f"Most important feature after training: **{top_feature['Feature']}** (Importance: {top_feature['Importance']})")
        st.dataframe(feature_importances_df.head(50), hide_index=True, width=800)

    if st.checkbox("Show Model Evaluation Metrics (slow)"):
        st.subheader("Model Evaluation Metrics")
        # Final Position Model
        X, y = get_features_and_target(data)
        mask = y.notnull() & np.isfinite(y)
        X, y = X[mask], y[mask]
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        avg_mse = -scores.mean()
        std_mse = scores.std()
        st.write(f"Final Position Model - Cross-validated MSE: {avg_mse:.3f} (± {std_mse:.3f})")

        # DNF Model
        X_dnf, y_dnf = get_features_and_target_dnf(data)
        mask_dnf = y_dnf.notnull() & np.isfinite(y_dnf)
        X_dnf, y_dnf = X_dnf[mask_dnf], y_dnf[mask_dnf]
        X_train_dnf, X_test_dnf, y_train_dnf, y_test_dnf = train_test_split(X_dnf, y_dnf, test_size=0.2, random_state=42)
        y_pred_dnf_proba = dnf_model.predict_proba(X_test_dnf)[:, 1]
        from sklearn.metrics import mean_absolute_error
        mae_dnf = mean_absolute_error(y_test_dnf, y_pred_dnf_proba)
        st.write(f"Mean Absolute Error (MAE) for DNF Probability (test set): {mae_dnf:.3f}")
        scores_dnf = cross_val_score(dnf_model, X_dnf, y_dnf, cv=5, scoring='roc_auc')
        st.write(f"DNF Model - Cross-validated ROC AUC: {scores_dnf.mean():.3f} (± {scores_dnf.std():.3f})")

        # Safety Car Model
        X_sc, y_sc = get_features_and_target_safety_car(safety_cars)
        mask_sc = y_sc.notnull() & np.isfinite(y_sc)
        X_sc, y_sc = X_sc[mask_sc], y_sc[mask_sc]
        scores_sc = cross_val_score(safetycar_model, X_sc, y_sc, cv=5, scoring='roc_auc')
        st.write(f"Safety Car Model - Cross-validated ROC AUC (unique rows): {scores_sc.mean():.3f} (± {scores_sc.std():.3f})")

    if st.checkbox("Show Safety Car Data Importances"):
        st.subheader("Safety Car Feature Importance")
        # Get feature names and importances from the trained safetycar_model
        preprocessor = safetycar_model.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
        importances = safetycar_model.named_steps['classifier'].coef_[0]

        # Odds ratio: exp(coef)
        odds_ratios = np.exp(importances)

        # Probability change (approximate, for small coefficients): sigmoid(coef) - 0.5
        prob_change = (1 / (1 + np.exp(-importances))) - 0.5

        df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': importances,
            'Odds Ratio': odds_ratios,
            'Prob Change (per unit)': prob_change
        }).sort_values('Coefficient', key=np.abs, ascending=False, ignore_index=True)

        st.dataframe(df, width=1000, hide_index=True)

    # --- Monte Carlo Feature Subset Search UI ---
    if st.checkbox("Run Monte Carlo Feature Subset Search"):
        st.subheader("Monte Carlo Feature Subset Search (Feature Selection)")

        # Get features and target from your data
        X, y = get_features_and_target(data)
        feature_names = X.columns.tolist()

        # User controls
        n_trials = st.number_input("Number of random trials", min_value=10, max_value=2000, value=50, step=10)
        min_features = st.number_input("Minimum features per trial", min_value=3, max_value=len(feature_names)-1, value=8, step=1)
        max_features = st.number_input(
        "Maximum features per trial",
        min_value=min_features+1,
        max_value=len(feature_names),
        value=min(min_features+1, len(feature_names)),
        step=1
    )
        
        # Run the Monte Carlo feature selection
        with st.spinner("Running Monte Carlo feature subset search..."):
            results = monte_carlo_feature_selection(
                X, y,
                model_class=lambda: XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist'),
                n_trials=int(n_trials),
                min_features=int(min_features),
                max_features=int(max_features),
                random_state=42
            )


        # Find the best feature sets
        results = sorted(results, key=lambda x: x['mae'])
        best = results[0]
        st.write("Best feature subset:", best['features'])
        st.write(", ".join([f"'{f}'" for f in best['features']]))
        st.write("Best MAE:", best['mae'])

        # Show top 20 feature sets
        st.subheader("Top 20 Feature Subsets")
        st.dataframe(pd.DataFrame(results[:20]), hide_index=True, column_config={
            "features": "Feature Subset",
            "mae": "Mean Absolute Error (MAE)"
        })

        # Count feature appearances in top 20 subsets
        from collections import Counter
        top_features = [f for r in results[:20] for f in r['features']]
        feature_counts = Counter(top_features)
        feature_counts_df = pd.DataFrame(feature_counts.items(), columns=['Feature', 'Appearances']).sort_values(by='Appearances', ascending=False)
        st.subheader("Feature Appearance in Top 20 Subsets")
        st.dataframe(feature_counts_df, hide_index=True, width=600)

    if st.checkbox("Run Recursive Feature Elimination (RFE)"):
        X, y = get_features_and_target(data)
        mask = y.notnull() & np.isfinite(y)
        X, y = X[mask], y[mask]
        # Convert object columns to category codes for RFE/XGBoost
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype('category').cat.codes
        
        n_features = st.number_input("Number of features to select (RFE)", min_value=1, max_value=len(X.columns), value=10, step=1)
        with st.spinner("Running RFE..."):
            selected_features, ranking = run_rfe_feature_selection(X, y, n_features_to_select=int(n_features))
        st.write("Selected features:", selected_features)
        st.dataframe(pd.DataFrame({'Feature': X.columns, 'Ranking': ranking}).sort_values('Ranking'), width=600, hide_index=True)

    if st.checkbox("Run Boruta Feature Selection"):
        X, y = get_features_and_target(data)
        mask = y.notnull() & np.isfinite(y)
        X, y = X[mask], y[mask]
        # Convert object columns to category codes for RFE/XGBoost
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype('category').cat.codes
        max_iter = st.number_input("Boruta max iterations", min_value=10, max_value=200, value=50, step=10)
        with st.spinner("Running Boruta..."):
            selected_features, ranking = run_boruta_feature_selection(X, y, max_iter=int(max_iter))
        st.write("Selected features:", selected_features)
        # Use the columns from the DataFrame used in Boruta
        st.dataframe(pd.DataFrame({'Feature': X.columns[:len(ranking)], 'Ranking': ranking}).sort_values('Ranking'))

    if st.checkbox("Run RFE to Minimize MAE"):
        X, y = get_features_and_target(data)
        mask = y.notnull() & np.isfinite(y)
        X, y = X[mask], y[mask]
        min_features = st.number_input("Min features", min_value=1, max_value=len(X.columns)-1, value=3, step=1)
        max_features = st.number_input("Max features", min_value=min_features+1, max_value=len(X.columns), value=10, step=1)
        with st.spinner("Running RFE to minimize MAE..."):
            best_features, best_ranking, best_mae, maes = rfe_minimize_mae(X, y, min_features=int(min_features), max_features=int(max_features))
        st.write(f"Best MAE: {best_mae:.3f}")
        st.write("Best feature subset:", best_features)
        st.dataframe(pd.DataFrame({'Feature': best_features}))
        st.line_chart(pd.DataFrame(maes, columns=['n_features', 'MAE']).set_index('n_features'))

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
                'numberOfStops': 'Number of Stops',
                'positionsGained': 'Positions Gained',
                'avgLapTime': 'Avg Lap Time',
                'finishingTime': 'Finishing Time',
            }
        )

        # Apply styling to highlight correlations
        correlation_matrix = correlation_matrix.style.map(highlight_correlation, subset=correlation_matrix.columns[1:])
        
        # Display the correlation matrix
        st.dataframe(correlation_matrix, column_config=correlation_columns_to_display, hide_index=True, height=600)

    if st.checkbox('Show Model Accuracy for All Races'):
        st.subheader("Model Accuracy Across All Races")

        # Extract features and target from the full dataset
        X, y = get_features_and_target(data)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model on the training set
        model, mse, r2, mae, mean_err, evals_result = train_and_evaluate_model(data)

        preprocessor = get_preprocessor_position()
        preprocessor.fit(X_train)  # Fit on training data
        X_test_prep = preprocessor.transform(X_test)
        y_pred = model.predict(xgb.DMatrix(X_test_prep))

        # Combine predictions and actuals for comparison
        results_df = X_test.copy()
        results_df['ActualFinalPosition'] = y_test.values
        results_df['PredictedFinalPosition'] = y_pred
        results_df['Error'] = results_df['ActualFinalPosition'] - results_df['PredictedFinalPosition']

        # Display metrics
        st.write(f"Mean Squared Error: {mse:.3f}")
        st.write(f"R^2 Score: {r2:.3f}")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"Mean Error: {mean_err:.2f}")

        # Show a table of predictions vs actuals
        st.dataframe(
            results_df[['grandPrixName', 'constructorName', 'resultsDriverName', 'ActualFinalPosition', 'PredictedFinalPosition', 'Error']].sort_values(by=['grandPrixName', 'ActualFinalPosition']),
            hide_index=True,
            width=1000,
            column_config={
                'grandPrixName': st.column_config.TextColumn("Grand Prix"),
                'constructorName': st.column_config.TextColumn("Constructor"),
                'resultsDriverName': st.column_config.TextColumn("Driver"),
                'ActualFinalPosition': st.column_config.NumberColumn("Actual", format="%d"),
                'PredictedFinalPosition': st.column_config.NumberColumn("Predicted", format="%.2f"),
                'Error': st.column_config.NumberColumn("Error", format="%.2f"),
            }
        )

        # Optional: Show a scatter plot of Actual vs Predicted
        st.subheader("Actual vs Predicted Final Position (All Races)")
        st.scatter_chart(results_df, x='ActualFinalPosition', y='PredictedFinalPosition', use_container_width=True)



    if st.checkbox('Show Raw Data'):

        st.write(f"Total number of results: {len(data):,d}")

        st.dataframe(data, column_config=columns_to_display,
            hide_index=True,  width=800, height=600)

    if st.checkbox("Run Hyperparameter Tuning"):
        X, y = get_features_and_target(data)
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [3, 4, 5],
            'regressor__learning_rate': [0.05, 0.1, 0.2],
            'regressor__reg_alpha': [0, 0.1, 0.3],           # L1 regularization
            'regressor__colsample_bytree': [0.6, 0.8, 1.0],  # Sample % of features per tree
            'regressor__colsample_bylevel': [0.6, 0.8, 1.0], # Sample % per tree level
            'regressor__colsample_bynode': [0.6, 0.8, 1.0],  # Sample % per split
        }
        pipeline = Pipeline([
            ('preprocessor', get_preprocessor_position()),
            ('regressor', XGBRegressor(random_state=42))
        ])
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')

        mask = y.notnull() & np.isfinite(y)
        X_clean, y_clean = X[mask], y[mask]
        grid_search.fit(X_clean, y_clean)
        st.write("Best params:", grid_search.best_params_)
        
        