import fastf1
from fastf1.ergast import Ergast
import pandas as pd
import datetime
import json
from os import path
import os
import sys
import subprocess
import streamlit as st
import numpy as np
import warnings
# suppress pandas FutureWarning about silent downcasting on fillna; prefer
# explicit infer_objects where possible, otherwise silence the noisy warning
warnings.filterwarnings(
    "ignore",
    message=r"Downcasting object dtype arrays on \.fillna, \.ffill, \.bfill is deprecated",
    category=FutureWarning,
)
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
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingRegressor, StackingRegressor
import seaborn as sns
from xgboost import XGBRegressor
import shap
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Import the temporal leakage audit helper. The helper lives in `scripts/`.
# Try multiple import strategies to be robust when Streamlit changes sys.path.
import logging
# Debugging toggle: set environment variable F1_DEBUG=1 to enable detailed
# runtime diagnostics (prints shapes, feature lists, and model-reported feature counts).
DEBUG = os.environ.get('F1_DEBUG', '0') == '1'
logger = logging.getLogger('f1analysis')
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)

# Attempt to import audit_temporal_leakage with fallbacks
audit_temporal_leakage = None  # type: ignore
try:
    import audit_temporal_leakage  # type: ignore
except ModuleNotFoundError:
    try:
        # Add repository scripts directory to sys.path and retry
        SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), 'scripts')
        if SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, SCRIPTS_DIR)
        import audit_temporal_leakage  # type: ignore
    except Exception:
        try:
            # Try module-style import if repo root is on sys.path
            import scripts.audit_temporal_leakage as audit_temporal_leakage  # type: ignore
        except Exception:
            audit_temporal_leakage = None
            if DEBUG:
                logger.debug('audit_temporal_leakage module not found; continuing without audit helpers')

EarlyStopping = xgb.callback.EarlyStopping

from footer import add_betting_oracle_footer



DATA_DIR = 'data_files/'

# Cache version - increment this when preprocessor logic changes
CACHE_VERSION = "v2.4"  # Bumped to invalidate cache after adding preprocessor return value

# Preprocessor used when training the main position model. Set during training so
# prediction uses the exact same feature ordering and transforms (prevents
# feature-shape mismatches between training and prediction environments).
TRAINING_PREPROCESSOR = None

def debug_log(msg, obj=None):
    """Helper to emit diagnostics both to Streamlit UI and logs when DEBUG is enabled."""
    if not DEBUG:
        return
    try:
        # Streamlit-friendly display
        try:
            st.write(f"DEBUG: {msg}")
            if obj is not None:
                st.write(obj)
        except Exception:
            pass
        # Logger
        if obj is None:
            logger.debug(msg)
        else:
            logger.debug(f"%s -- %r", msg, obj)
    except Exception:
        pass

# Suppress numpy warnings about empty slices during calculations
warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning, module='numpy')
# Suppress noisy numpy divide/invalid value RuntimeWarnings caused by correlation/stddev ops
warnings.filterwarnings('ignore', message='invalid value encountered in divide', category=RuntimeWarning, module='numpy')
# Also set numpy to ignore invalid operations to avoid repetitive RuntimeWarnings during UI calculations
np.seterr(invalid='ignore')


def compute_safe_correlation(full_df, cols, method='pearson'):
    """Compute correlation for `cols` from `full_df`, dropping constant or all-NaN columns.

    Returns a square DataFrame indexed/columned by `cols`. Columns that were constant
    or all-NaN will be present but filled with NaN so downstream code that expects
    a fixed shape can still rename rows/columns safely.
    """
    # Defensive: ensure cols exist and deduplicate while preserving order
    cols = [c for c in cols if c in full_df.columns]
    if not cols:
        return pd.DataFrame()

    seen = set()
    cols_unique = []
    for c in cols:
        if c not in seen:
            cols_unique.append(c)
            seen.add(c)
    # use the deduplicated ordered list for downstream operations
    cols = cols_unique

    sub = full_df[cols]
    # select numeric columns for correlation
    num = sub.select_dtypes(include=[np.number])

    # columns with more than one unique non-null value
    # Use pd.unique on the dropped-NA values to ensure we get a concrete length
    # (avoids ambiguous truth values if nunique ever returns a non-scalar)
    keep_cols = [
        c for c in num.columns
        if len(pd.unique(num[c].dropna())) > 1
    ]

    # compute correlation only on keep_cols
    if keep_cols:
        with np.errstate(invalid='ignore', divide='ignore'):
            corr_partial = num[keep_cols].corr(method=method)
    else:
        corr_partial = pd.DataFrame()

    # build a full square matrix with original cols, fill with NaN
    full_corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
    if not corr_partial.empty:
        # place partial results into full matrix for the kept cols
        for r in corr_partial.index:
            for c in corr_partial.columns:
                full_corr.at[r, c] = corr_partial.at[r, c]

    return full_corr

def create_constructor_adjusted_driver_features(data):
    """
    Create driver performance features that are adjusted by constructor performance.
    This helps account for drivers who have changed teams.
    """
    try:
        # Check if required columns exist
        required_cols = ['grandPrixYear', 'constructorName', 'resultsFinalPositionNumber']
        if not all(col in data.columns for col in required_cols):
            return data
            
        # Handle Points column (could have different names)
        points_col = None
        for col in ['Points', 'Points_results_with_qualifying', 'points']:
            if col in data.columns:
                points_col = col
                break
        
        # Handle podium column
        podium_col = None
        for col in ['resultsPodium', 'podium', 'Podium']:
            if col in data.columns:
                podium_col = col
                break
        
        # Calculate constructor performance by year
        agg_dict = {'resultsFinalPositionNumber': 'mean'}
        col_names = ['grandPrixYear', 'constructorName', 'constructorAvgPosition']
        
        if points_col:
            agg_dict[points_col] = 'mean'
            col_names.append('constructorAvgPoints')
            
        if podium_col:
            agg_dict[podium_col] = 'mean'
            col_names.append('constructorPodiumRate')
        
        constructor_performance = data.groupby(['grandPrixYear', 'constructorName']).agg(agg_dict).reset_index()
        constructor_performance.columns = col_names
        
        # Merge back to main data
        data_enhanced = data.merge(constructor_performance, on=['grandPrixYear', 'constructorName'], how='left')
        
        # Create relative driver performance metrics
        data_enhanced['driverVsConstructorPosition'] = data_enhanced['resultsFinalPositionNumber'] - data_enhanced['constructorAvgPosition']
        
        if points_col and 'constructorAvgPoints' in data_enhanced.columns:
            data_enhanced['driverRelativeToConstructor'] = data_enhanced[points_col] / (data_enhanced['constructorAvgPoints'] + 0.1)
        
        return data_enhanced
        
    except Exception as e:
        return data

def create_recent_performance_features(data, recent_races=5):
    """
    Create features based on recent performance to weight newer data more heavily.
    """
    try:
        # Check if required columns exist
        required_cols = ['resultsDriverId', 'grandPrixYear', 'resultsFinalPositionNumber']
        if not all(col in data.columns for col in required_cols):
            return data
            
        # Check for round column (might have different names)
        round_col = None
        for col in ['round', 'Round', 'race_round', 'grandPrixRound']:
            if col in data.columns:
                round_col = col
                break
        
        if not round_col:
            return data
            
        # Handle Points column
        points_col = None
        for col in ['Points', 'Points_results_with_qualifying', 'points']:
            if col in data.columns:
                points_col = col
                break
        
        data_sorted = data.sort_values(['resultsDriverId', 'grandPrixYear', round_col]).copy()
        
        # Calculate rolling averages for recent performance
        for window in [3, 5, 10]:
            data_sorted[f'recentAvgPosition_{window}'] = (
                data_sorted.groupby('resultsDriverId')['resultsFinalPositionNumber']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            if points_col:
                data_sorted[f'recentAvgPoints_{window}'] = (
                    data_sorted.groupby('resultsDriverId')[points_col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
        
        return data_sorted
        
    except Exception as e:
        return data

def create_constructor_compatibility_features(data):
    """
    Create features that measure how well a driver performs with their current constructor
    vs their career average.
    """
    try:
        # Check if required columns exist
        required_cols = ['resultsDriverId', 'constructorName', 'resultsFinalPositionNumber']
        if not all(col in data.columns for col in required_cols):
            return data
            
        # Handle Points and podium columns
        points_col = None
        for col in ['Points', 'Points_results_with_qualifying', 'points']:
            if col in data.columns:
                points_col = col
                break
                
        podium_col = None
        for col in ['resultsPodium', 'podium', 'Podium']:
            if col in data.columns:
                podium_col = col
                break
        
        # Driver's career average
        career_agg = {'resultsFinalPositionNumber': 'mean'}
        career_cols = ['resultsDriverId', 'driverCareerAvgPosition']
        
        if points_col:
            career_agg[points_col] = 'mean'
            career_cols.append('driverCareerAvgPoints')
            
        if podium_col:
            career_agg[podium_col] = 'mean'
            career_cols.append('driverCareerPodiumRate')
        
        driver_career_avg = data.groupby('resultsDriverId').agg(career_agg).reset_index()
        driver_career_avg.columns = career_cols
        
        # Driver's performance with current constructor
        constructor_agg = {
            'resultsFinalPositionNumber': 'mean',
            'grandPrixYear': 'count'  # Number of races with this constructor
        }
        constructor_cols = ['resultsDriverId', 'constructorName', 'driverConstructorAvgPosition', 'racesWithConstructor']
        
        if points_col:
            constructor_agg[points_col] = 'mean'
            constructor_cols.insert(-1, 'driverConstructorAvgPoints')
            
        if podium_col:
            constructor_agg[podium_col] = 'mean'
            constructor_cols.insert(-1, 'driverConstructorPodiumRate')
        
        driver_constructor_performance = data.groupby(['resultsDriverId', 'constructorName']).agg(constructor_agg).reset_index()
        driver_constructor_performance.columns = constructor_cols
        
        # Merge features
        data_enhanced = data.merge(driver_career_avg, on='resultsDriverId', how='left')
        data_enhanced = data_enhanced.merge(driver_constructor_performance, on=['resultsDriverId', 'constructorName'], how='left')
        
        # Create compatibility metrics
        if 'driverCareerAvgPosition' in data_enhanced.columns and 'driverConstructorAvgPosition' in data_enhanced.columns:
            data_enhanced['constructorCompatibilityPosition'] = data_enhanced['driverCareerAvgPosition'] - data_enhanced['driverConstructorAvgPosition']
        
        if points_col and 'driverCareerAvgPoints' in data_enhanced.columns and 'driverConstructorAvgPoints' in data_enhanced.columns:
            data_enhanced['constructorCompatibilityPoints'] = data_enhanced['driverConstructorAvgPoints'] / (data_enhanced['driverCareerAvgPoints'] + 0.1)
        
        # Weight by experience with constructor (more races = more reliable metric)
        if 'racesWithConstructor' in data_enhanced.columns:
            data_enhanced['constructorExperienceWeight'] = np.clip(data_enhanced['racesWithConstructor'] / 10, 0.1, 1.0)
        
        return data_enhanced
        
    except Exception as e:
        return data

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
   page_title="Gridlocked - Formula 1 Betting & Analytics",
   page_icon=path.join(DATA_DIR, 'favicon.png'),
   layout="wide",
   initial_sidebar_state="expanded"
)

def km_to_miles(km):
    return km * 0.621371

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

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
    'constructorTotalPolePositions': 'Total Pole Positions (Constructor)',
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
    'Points': 'Current Year Points (Driver)',
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
    'totalRaceLaps': 'Total Race Laps (Constructor)',
    'totalPodiums': 'Total Podiums (Constructor)',
    'totalPodiumRaces': 'Total Podium Races (Constructor)',
    'totalPoints' : 'Total Points (Lifetime)',
    'totalChampionshipPoints': 'Total Champ Points',
    # 'totalPolePositions' : 'Total Pole Positions',
    'totalFastestLaps': 'Total Fastest Laps',
    # 'bestQualifyingTime_sec': 'Best Qualifying Time (s)',
    # 'delta_from_race_avg': 'Delta from Race Avg. (s)',
    'driverAge': 'Driver Age',
    'currentRookie': 'Current Rookie',
    'championship_position': 'Current Championship Position',
    # 'totalPoints': 'Current Year Points'
    'practice_x_safetycar_bin': 'Practice x Safety Car %', 
    'positions_gained_first_lap_pct_bin': 'Positions Gained First Lap %', 
    'is_first_season_with_constructor': 'First Season with Constructor', 
    'grid_penalty_x_constructor_bin': 'Grid Penalty x Constructor', 
    'SafetyCarStatus': 'Safety Car Status',
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
    'PredictedFinalPosition_Low': st.column_config.NumberColumn("Final Pos (Low)", format="%.3f"),
    'PredictedFinalPosition_High': st.column_config.NumberColumn("Final Pos (High)", format="%.3f"),
    'PredictedPositionMAE': st.column_config.NumberColumn("Position MAE", format="%.3f"),
    'PredictedPositionMAE_Low': st.column_config.NumberColumn("Position MAE (Low)", format="%.3f"),
    'PredictedPositionMAE_High': st.column_config.NumberColumn("Position MAE (High)", format="%.3f"),
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
def load_correlation(nrows, CACHE_VERSION):
    correlation_matrix = pd.read_csv(path.join(DATA_DIR, 'f1PositionCorrelation.csv'), sep='\t', nrows=nrows)
    return correlation_matrix

correlation_matrix = load_correlation(10000, CACHE_VERSION)

@st.cache_data
def load_data_schedule(nrows, CACHE_VERSION):
    raceSchedule = pd.read_json(path.join(DATA_DIR, 'f1db-races.json'))
    grandPrix = pd.read_json(path.join(DATA_DIR, 'f1db-grands-prix.json'))
    raceSchedule = raceSchedule.merge(grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_grandPrix', '_schedule'])
    #raceSchedule = raceSchedule.merge(grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_grandPrix', '_schedule'])
    return raceSchedule

raceSchedule = load_data_schedule(10000, CACHE_VERSION)

@st.cache_data
def load_drivers(nrows, CACHE_VERSION):
    drivers = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json'))
    return drivers

drivers = load_drivers(10000, CACHE_VERSION)

@st.cache_data
def load_qualifying(nrows):
    # Include cache version to invalidate when preprocessor changes
    _ = CACHE_VERSION
    qualifying = pd.read_csv(path.join(DATA_DIR, 'all_qualifying_races.csv'), sep='\t')
    return qualifying

qualifying = load_qualifying(10000)

@st.cache_data
def load_practices(nrows, CACHE_VERSION):
    practices = pd.read_csv(path.join(DATA_DIR, 'all_practice_laps.csv'), sep='\t', dtype={'PitOutTime': str}) 
    practices = practices[practices['Driver'] != 'ERROR']  # Remove rows where Driver is 'ERROR'
    return practices

practices = load_practices(10000, CACHE_VERSION)

@st.cache_data
def load_data_race_messages(nrows, CACHE_VERSION):
    race_messages = pd.read_csv(path.join(DATA_DIR, 'race_control_messages_grouped_with_dnf.csv'),sep='\t')
    return race_messages

race_messages = load_data_race_messages(10000, CACHE_VERSION)


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
def load_weather_data(nrows, CACHE_VERSION):
    weather = pd.read_csv(path.join(DATA_DIR, 'f1WeatherData_Grouped.csv'), sep='\t', nrows=nrows, usecols=['grandPrixId', 'short_date', 'average_temp', 'total_precipitation', 'average_humidity', 'average_wind_speed', 'id_races'])
    grandPrix = pd.read_json(path.join(DATA_DIR, 'f1db-grands-prix.json'))
    weather_with_grandprix = pd.merge(weather, grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_weather', '_grandPrix'])
    return weather_with_grandprix

weatherData = load_weather_data(10000, CACHE_VERSION)


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

# Page title and logo
st.image(path.join(DATA_DIR, 'gridlocked-logo-with-text.png'), width=450)
st.title(f'F1 Races from {raceNoEarlierThan} to {current_year}')
st.caption(f"Last updated: {readable_time}")
st.caption(f"Code deployed at: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

# Table styling toggle - set to False to revert to default borders
CLEAN_TABLE_BORDERS = True

# Table styling note: st.dataframe() renders in an isolated iframe
# so custom CSS cannot style its internal borders. You can style the container
# but not individual cells/borders. To customize table appearance fully,
# use st.table() (static), HTML tables, or third-party components like AgGrid.

if CLEAN_TABLE_BORDERS:
    st.markdown("""
    <style>
        /* Style the dataframe container only - internals are isolated */
        div[data-testid="stDataFrame"] {
            border: none !important;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

# Create main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Explorer", 
    "📈 Analytics & Visualizations", 
    "🏎️ Schedule",
    "🏁 Next Race",
    "🤖 Predictive Models",
    "💾 Data & Debug"
])

columns_to_display = {
    'grandPrixYear': st.column_config.NumberColumn("Year", format="%d"),
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
    'currentRookie': st.column_config.CheckboxColumn("Rookie"),
    'championship_position': st.column_config.NumberColumn(
        "Championship Position", format="%d", min_value=0, max_value=100, step=1, default=0), 

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
    'courseLength': st.column_config.NumberColumn("Lap Length (km)", format="%.2f"),
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
def load_data(nrows, CACHE_VERSION):
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
              'qualifying_consistency_std', 'driver_starting_position_3_races', 'driver_starting_position_5_races', 'abbreviation', 
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
                                        'qualifying_position_vs_field_recent_form', 'currentRookie', 'driver_constructor_id',
                                        'podium_form_3_races', 'wins_last_5_races', 'championship_position', 'points_leader_gap',
                                        'pole_to_win_rate', 'front_row_conversion', 'recent_wins_3_races',
                                        'rolling_3_race_win_percentage', 'recent_qualifying_improvement_trend', 'head_to_head_teammate_performance_delta', 'championship_position_pressure_factor',
                                        'constructor_recent_mechanical_dnf_rate', 'driver_performance_at_circuit_type', 'weather_pattern_analysis_by_location', 'overtaking_difficulty_index',
                                        'q1_q2_q3_sector_consistency', 'qualifying_position_vs_race_pace_delta_by_track']

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

    return fullResults, pitStops

data, pitStops = load_data(10000, CACHE_VERSION)

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

# Apply team-aware feature engineering (after column renaming)
try:
    data = create_constructor_adjusted_driver_features(data)
    data = create_recent_performance_features(data, recent_races=5)
    data = create_constructor_compatibility_features(data)
except Exception as e:
    st.warning(f"Could not create some team-aware features: {e}")
    # Continue with original data if feature creation fails

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
data['Points'] = data['Points'].astype('Int64')
data['driverRank'] = data['driverRank'].astype('Int64')
# if 'bestQualifyingTime_sec' in data.columns:
#     data['bestQualifyingTime_sec'] = data['bestQualifyingTime_sec'].astype('Float64')
# else:
#     st.warning("'bestQualifyingTime_sec' column not found in data.")
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
# data['totalPolePositions'] = data['totalPolePositions_results_with_qualifying'].astype('Int64')
data['totalFastestLaps'] = data['totalFastestLaps_results_with_qualifying'].astype('Int64')
data['totalRaceEntries'] = data['totalRaceEntries_results_with_qualifying'].astype('Int64')
data['driverAge'] = data['driverAge'].astype('Int64')
# data['delta_from_race_avg'] = data['delta_from_race_avg'].astype('Float64')
data['driverAge'] = data['driverAge'].astype('Int64')
data['DNF'] = data['DNF'].astype('boolean')
data['championship_position'] = data['championship_position'].astype('Float64')
data['practice_x_safetycar_bin'] = data['practice_x_safetycar_bin'].astype('Float64')
data['positions_gained_first_lap_pct_bin'] = data['positions_gained_first_lap_pct_bin'].astype('Float64')
data['is_first_season_with_constructor'] = data['is_first_season_with_constructor'].astype('boolean')
data['grid_penalty_x_constructor_bin'] = data['grid_penalty_x_constructor_bin'].astype('Float64')
data['SafetyCarStatus'] = data['SafetyCarStatus'].astype('Float64')

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
                  'CleanAirAvg_FP2', 'DirtyAirAvg_FP2', 'Delta_FP2', 'CleanAirAvg_FP3', 'DirtyAirAvg_FP3','Delta_FP3', 'SafetyCarStatus', 'finishing_position_std_driver', 'finishing_position_std_constructor',
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
                                        'qualifying_position_vs_field_recent_form', #'driver_constructor_id', 'qualifying_position_vs_teammate_historical'       
                                        'driverConstructorAvgPosition', 'driverConstructorAvgPoints', 'driverConstructorPodiumRate', 'racesWithConstructor', 'constructorCompatibilityPosition', 'constructorCompatibilityPoints', 'constructorExperienceWeight', 
                                        'driverVsConstructorPosition', 'driverRelativeToConstructor', 'recentAvgPosition_3', 'recentAvgPoints_3', 'recentAvgPosition_5', 'recentAvgPoints_5', 'recentAvgPosition_10', 'recentAvgPoints_10', 
                                        'points_leader_gap', 'pole_to_win_rate', 'front_row_conversion', 'recent_wins_3_races', 'rolling_3_race_win_percentage', 'recent_qualifying_improvement_trend', 'head_to_head_teammate_performance_delta', 
                                        'championship_position_pressure_factor', 'constructor_recent_mechanical_dnf_rate', 'driver_performance_at_circuit_type', 'weather_pattern_analysis_by_location', 
                                        'overtaking_difficulty_index', 'q1_q2_q3_sector_consistency', 'qualifying_position_vs_race_pace_delta_by_track', 'numberOfStops', 
                                        'driverCareerAvgPosition', 'driverCareerAvgPoints', 'driverCareerPodiumRate', 'driver_constructor_id', 'qualifying_position_vs_teammate_historical', 
                                        'podium_form_3_races', 'wins_last_5_races', 'constructorAvgPosition', 'constructorAvgPoints', 'constructorPodiumRate', 'totalPoints', 'totalFastestLaps', 'totalPolePositions',
                                        

        ]

suffixes_to_exclude = ('_x', '_y', '_qualifying', '_results_with_qualifying', '_drivers', '_mph', '_sec', '.1', '.2', '.3', '_bin')
auto_exclusions = [col for col in column_names if col.endswith(suffixes_to_exclude)]
exclusionList = exclusionList + auto_exclusions

# If errant/extra columns appear on the left in filters, the below analysis will point them out.

# st.write(f"Exclusion List: {exclusionList}")

# remaining_columns = [col for col in column_names if col not in exclusionList]
# st.write(f"Remaining Columns: {remaining_columns}")

column_names.sort()

# all of the non-leaky fields from the fullResults dataset (9/19/2025)
def get_features_and_target(data):
    
    features = [
        # 'grandPrixName',  # Removed categorical feature to avoid imputation errors
        'resultsDriverName',
        'constructorName',
        'resultsStartingGridPositionNumber',
        'lastFPPositionNumber',
        'resultsQualificationPositionNumber',
        'grandPrixLaps',
        'constructorTotalRaceStarts',
        'activeDriver',
        'recent_form_5_races_bin',
        'yearsActive',
        'driverDNFAvg',
        'best_s1_sec_bin',
        'LapTime_sec_bin',
        'SpeedI2_mph_bin',
        'SpeedST_mph_bin',
        'constructor_recent_form_3_races_bin',
        'constructor_recent_form_5_races_bin',
        'CleanAirAvg_FP1_bin',
        'Delta_FP1_bin',
        'DirtyAirAvg_FP2_bin',
        'Delta_FP2_bin',
        'Delta_FP3_bin',
        'engineManufacturerId',
        'delta_from_race_avg_bin',
        'driverAge',
        'finishing_position_std_driver',
        'finishing_position_std_constructor',
        'delta_lap_2_historical',
        'delta_lap_10_historical',
        'delta_lap_15_historical',
        'delta_lap_20_historical',
        'driver_dnf_rate_5_races',
        'avg_final_position_per_track_bin',
        'last_final_position_per_track',
        'avg_final_position_per_track_constructor_bin',
        'practice_position_improvement_1P_2P',
        'practice_position_improvement_2P_3P',
        'practice_position_improvement_1P_3P',
        'practice_time_improvement_1T_2T_bin',
        'practice_time_improvement_time_time_bin',
        'teammate_practice_delta_bin',
        'last_final_position_per_track_constructor',
        'driver_starting_position_3_races_bin',
        'qualPos_x_last_practicePos_bin',
        'qualPos_x_avg_practicePos_bin',
        'recent_form_median_3_races',
        'recent_form_median_5_races',
        'recent_form_worst_3_races',
        'recent_positions_gained_3_races_bin',
        'driver_positionsGained_3_races_bin',
        'qual_vs_track_avg_bin',
        'constructor_avg_practice_position_bin',
        'practice_position_std_bin',
        'recent_vs_season_bin',
        'practice_improvement',
        'qual_x_constructor_wins_bin',
        'grid_penalty',
        'grid_penalty_x_constructor_bin',
        'recent_form_x_qual_bin',
        'driver_rank_x_constructor_rank',
        'practice_gap_to_teammate_bin',
        'street_experience',
        'fp1_lap_delta_vs_best_bin',
        'last_race_vs_track_avg_bin',
        'top_speed_rank_bin',
        'historical_avgLapPace_bin',
        'pit_delta_x_driver_age_bin',
        'constructor_points_x_grid_bin',
        'dnf_rate_x_practice_std_bin',
        'grid_penalty_x_constructor_rank',
        'constructor_win_rate_3y',
        'driver_podium_rate_3y',
        'track_familiarity',
        'recent_podium_streak',
        'grid_position_percentile_bin',
        # 'constructor_recent_win_streak',
        'qual_to_final_delta_5yr_bin',
        'qual_to_final_delta_3yr_bin',
        'overtake_potential_3yr_bin',
        'overtake_potential_5yr_bin',
        'constructor_avg_qual_pos_at_track_bin',
        'driver_avg_grid_pos_at_track_bin',
        'driver_avg_practice_pos_at_track_bin',
        'constructor_avg_practice_pos_at_track_bin',
        'constructor_qual_improvement_3r_bin',
        'constructor_practice_improvement_3r_bin',
        'driver_teammate_qual_gap_3r_bin',
        'driver_teammate_practice_gap_3r_bin',
        'driver_street_qual_avg',
        'driver_track_qual_avg',
        # 'driver_street_practice_avg',
        'driver_high_wind_qual_avg',
        'driver_high_humidity_qual_avg',
        'driver_wet_qual_avg',
        'driver_safetycar_qual_avg',
        'driver_safetycar_practice_avg',
        'races_with_constructor_bin',
        'driver_constructor_avg_final_position_bin',
        'constructor_dnf_rate_3_races',
        'constructor_dnf_rate_5_races',
        'historical_race_pace_vs_median_bin',
        'practice_consistency_vs_teammate_bin',
        'fp3_position_percentile_bin',
        'constructor_practice_improvement_rate_bin',
        'track_fp1_fp3_improvement_bin',
        'teammate_practice_delta_at_track_bin',
        'qual_vs_track_median',
        'qual_improvement_vs_field_avg_bin',
        'driver_podium_rate_at_track',
        'fp3_vs_constructor_avg_bin',
        'qual_vs_constructor_avg_bin',
        'practice_lap_time_consistency_bin',
        'qual_lap_time_consistency_bin',
        'practice_improvement_vs_teammate_bin',
        'qual_improvement_vs_teammate_bin',
        'practice_vs_best_at_track_bin',
        'qual_vs_best_at_track',
        'qual_vs_worst_at_track',
        'practice_position_percentile_vs_constructor_bin',
        'qualifying_position_percentile_vs_constructor_bin',
        'practice_lap_time_delta_to_constructor_best_bin',
        'qualifying_lap_time_delta_to_constructor_best_bin',
        'qualifying_position_vs_field_best_at_track',
        'practice_position_vs_field_worst_at_track_bin',
        'qualifying_position_vs_field_worst_at_track',
        'qualifying_position_vs_field_median_at_track',
        'practice_position_vs_constructor_best_at_track_bin',
        'qualifying_position_vs_constructor_best_at_track',
        'qualifying_position_vs_constructor_worst_at_track',
        'practice_position_vs_constructor_median_at_track_bin',
        'practice_lap_time_consistency_vs_field_bin',
        'qualifying_lap_time_consistency_vs_field_bin',
        'practice_position_vs_field_recent_form_bin',
        'qualifying_position_vs_field_recent_form_bin',
        'podium_form_3_races', 'wins_last_5_races', 'championship_position', 'points_leader_gap',
        'pole_to_win_rate', 'front_row_conversion', 'recent_wins_3_races',
        'rolling_3_race_win_percentage', 'recent_qualifying_improvement_trend', 'head_to_head_teammate_performance_delta', 'championship_position_pressure_factor',
        'constructor_recent_mechanical_dnf_rate', 'driver_performance_at_circuit_type', 'weather_pattern_analysis_by_location', 'overtaking_difficulty_index',
        'q1_q2_q3_sector_consistency', 'qualifying_position_vs_race_pace_delta_by_track'
    ]
    
    # Add team-aware features if they exist (for drivers who change constructors)
    team_aware_features = [
        'constructorAvgPosition', 'constructorPodiumRate', 'constructorAvgPoints',
        'driverVsConstructorPosition', 'driverRelativeToConstructor',
        'recentAvgPosition_3', 'recentAvgPosition_5', 'recentAvgPosition_10',
        'recentAvgPoints_3', 'recentAvgPoints_5', 'recentAvgPoints_10',
        'driverCareerAvgPosition', 'driverCareerAvgPoints', 'driverCareerPodiumRate',
        'driverConstructorAvgPosition', 'driverConstructorAvgPoints', 'driverConstructorPodiumRate',
        'racesWithConstructor', 'constructorCompatibilityPosition', 'constructorCompatibilityPoints',
        'constructorExperienceWeight'
    ]
    
    # Only add team-aware features that actually exist in the data
    available_team_features = [f for f in team_aware_features if f in data.columns]
    features.extend(available_team_features)
    

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

# Known categorical features
categorical_features_known = [
    'grandPrixName',
    'resultsDriverName',
    'engineManufacturerId',
    'constructorName',
    'driverName',
    'circuitName',
    'circuitCountry',
    'circuitLocation',
    'nationality',
    'driverNationality',
    'constructorNationality'
]

# Function to load features from text file
def load_f1_position_model_features():
    filepath = 'data_files/f1_position_model_numerical_features.txt'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            default_numerical = [line.strip() for line in f if line.strip()]
    else:
        default_numerical = []
    
    monte_carlo_filepath = 'data_files/f1_position_model_best_features_monte_carlo.txt'
    if os.path.exists(monte_carlo_filepath):
        with open(monte_carlo_filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('Best MAE')]
        # Separate into numerical and categorical
        numerical = [f for f in lines if f in default_numerical]
        categorical = []  # Temporarily disable categorical to avoid MAE increase
        if numerical or categorical:
            
            return numerical, categorical
    
    return default_numerical, []

numerical_features, categorical_features = load_f1_position_model_features()

def get_preprocessor_position(X=None):
    global numerical_features, categorical_features
    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # If no features were loaded from files, fall back to all available features
    if not numerical_features and not categorical_features:
        # Get all features from the data and determine which are numerical vs categorical
        if X is None:
            # This shouldn't happen in normal flow, but provide a fallback
            numerical_features_fallback = []
            categorical_features_fallback = []
        else:
            from pandas.api.types import is_numeric_dtype
            numerical_features_fallback = [col for col in X.columns if is_numeric_dtype(X[col])]
            categorical_features_fallback = [col for col in X.columns if not is_numeric_dtype(X[col])]
        
        transformers = [
            ('num', Pipeline(steps=[
                ('imputer', numerical_imputer),
                ('scaler', StandardScaler())
            ]), numerical_features_fallback)
        ]
        
        if categorical_features_fallback:
            transformers.append((
                'cat', Pipeline(steps=[
                    ('imputer', categorical_imputer),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features_fallback
            ))
    else:
        # Filter out numerical features that are all NaN to avoid sklearn imputation warnings
        if X is not None:
            numerical_features = [col for col in numerical_features if col in X.columns and not X[col].isna().all()]
        
        transformers = [
            ('num', Pipeline(steps=[
                ('imputer', numerical_imputer),
                ('scaler', StandardScaler())
            ]), numerical_features)
        ]
        
        if categorical_features:
            transformers.append((
                'cat', Pipeline(steps=[
                    ('imputer', categorical_imputer),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features
            ))

    preprocessor = ColumnTransformer(transformers=transformers)
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
def load_safetycars(nrows, CACHE_VERSION):
    safety_cars = pd.read_csv(path.join(DATA_DIR, 'f1SafetyCarFeatures.csv'), sep='\t', nrows=nrows)
    safety_cars = safety_cars.drop_duplicates()
    # Drop duplicate rows based on all feature columns
    features, _ = get_features_and_target_safety_car(safety_cars)
    safety_cars = safety_cars.drop_duplicates(subset=features.columns.tolist())
    return safety_cars
safety_cars = load_safetycars(10000, CACHE_VERSION)


###### Training model for final racing position prediction

features, _ = get_features_and_target(data)
missing = [col for col in features.columns if col not in data.columns]
if missing:
    st.error(f"The following feature columns are missing from your data: {missing}")
    st.stop()


def train_and_evaluate_model(data, early_stopping_rounds=20, model_type="XGBoost", preprocessor_version="v2"):
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    import numpy as np

    X, y = get_features_and_target(data)
    preprocessor = get_preprocessor_position(X)

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

     # ADD THIS: Create sample weights favoring top positions
    # sample_weights_train = np.where(y_train <= 3, 3.0,     # 3x weight for podium
    #                       np.where(y_train <= 10, 2.0,     # 2x weight for points
    #                               1.0))                     # Normal weight for others
    # In your train_and_evaluate_model function, try lighter weights:
    sample_weights_train = np.where(y_train == 1, 2.0,      # 2x weight for winners only
                      np.where(y_train <= 3, 1.5,       # 1.5x weight for podium
                      np.where(y_train <= 10, 1.2,      # 1.2x weight for points
                              1.0)))

    # Preprocess manually and store the fitted preprocessor globally so prediction
    # uses identical feature ordering and transforms (important on Streamlit Cloud).
    global TRAINING_PREPROCESSOR
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    TRAINING_PREPROCESSOR = preprocessor

    if model_type == "XGBoost":
        # Use XGBRegressor for better compatibility with early stopping on Streamlit Cloud
        
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric='mae'
        )
        
        model.fit(
            X_train_prep, y_train,
            sample_weight=sample_weights_train,
            eval_set=[(X_test_prep, y_test)],
            verbose=False
        )
        
        # Get evaluation results
        evals_result = {}
        if hasattr(model, 'evals_result_'):
            evals_result = model.evals_result_
        else:
            # Fallback for older versions
            evals_result = {'eval': {'mae': [getattr(model, 'best_score', 0)]}}
        
        # Predict
        y_pred = model.predict(X_test_prep)
        
        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mean_err = np.mean(y_pred - y_test)
        
        return model, mse, r2, mae, mean_err, evals_result, preprocessor

    elif model_type == "LightGBM":
        from lightgbm import LGBMRegressor
        
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train_prep, y_train,
            sample_weight=sample_weights_train,
            eval_set=[(X_test_prep, y_test)],
            eval_metric='mae'
        )
        
        y_pred = model.predict(X_test_prep)
        
        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mean_err = np.mean(y_pred - y_test)
        
        # Mock evals_result
        evals_result = {'eval': {'mae': [mae]}}
        
        return model, mse, r2, mae, mean_err, evals_result, preprocessor

    elif model_type == "CatBoost":
        from catboost import CatBoostRegressor
        
        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=4,
            random_state=42,
            verbose=False,
            early_stopping_rounds=early_stopping_rounds
        )
        
        model.fit(
            X_train_prep, y_train,
            sample_weight=sample_weights_train,
            eval_set=[(X_test_prep, y_test)],
            verbose=False
        )
        
        y_pred = model.predict(X_test_prep)
        
        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mean_err = np.mean(y_pred - y_test)
        
        # Mock evals_result
        evals_result = {'eval': {'mae': [mae]}}
        
        return model, mse, r2, mae, mean_err, evals_result, preprocessor

    elif model_type == "Ensemble (XGBoost + LightGBM + CatBoost)":
        from sklearn.ensemble import StackingRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor
        
        base_estimators = [
            ('xgb', XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)),
            ('lgb', LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)),
            ('cat', CatBoostRegressor(iterations=100, depth=4, learning_rate=0.1, random_state=42, verbose=False))
        ]
        
        model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1),
            cv=3,
            n_jobs=-1
        )
        
        model.fit(X_train_prep, y_train, sample_weight=sample_weights_train)
        y_pred = model.predict(X_test_prep)
        
        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mean_err = np.mean(y_pred - y_test)
        
        # Mock evals_result
        evals_result = {'eval': {'mae': [mae]}}
        
        return model, mse, r2, mae, mean_err, evals_result, preprocessor


@st.cache_data
def train_and_evaluate_dnf_model(data, CACHE_VERSION):
    from sklearn.linear_model import LogisticRegression
    X, y = get_features_and_target_dnf(data)
    preprocessor = get_preprocessor_dnf()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model


@st.cache_data
def train_and_evaluate_safetycar_model(data, CACHE_VERSION):
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

def load_pretrained_model(model_name='position_model', CACHE_VERSION='v2.3'):
    """Load a pre-trained model from disk if available."""
    import pickle
    from pathlib import Path
    
    models_dir = Path('data_files/models')
    model_file = models_dir / f'{model_name}.pkl'
    
    if model_file.exists():
        try:
            with open(model_file, 'rb') as f:
                artifact = pickle.load(f)
            
            # Check cache version compatibility
            if artifact.get('cache_version') == CACHE_VERSION:
                return artifact
            else:
                st.info(f"Pre-trained {model_name} found but cache version mismatch. Will retrain.")
        except Exception as e:
            st.warning(f"Error loading pre-trained {model_name}: {e}. Will retrain.")
    
    return None

@st.cache_data
def get_trained_model(early_stopping_rounds, CACHE_VERSION, force_retrain=False):
    """Load or train the position prediction model. Returns (model, mse, r2, mae, mean_err, evals_result, preprocessor)."""
    # Try to load pre-trained model first
    if not force_retrain:
        pretrained = load_pretrained_model('position_model', CACHE_VERSION)
        if pretrained is not None:
            return (pretrained['model'], pretrained['mse'], pretrained['r2'], 
                    pretrained['mae'], pretrained['mean_err'], pretrained['evals_result'],
                    pretrained.get('preprocessor'))
    
    # Fall back to training
    model, mse, r2, mae, mean_err, evals_result, preprocessor = train_and_evaluate_model(data, early_stopping_rounds=early_stopping_rounds)
    return model, mse, r2, mae, mean_err, evals_result, preprocessor

# Lazy-load models (only when accessed, not at module load)
def get_main_model():
    if 'main_model' not in st.session_state:
        model, mse, r2, mae, mean_err, evals_result, preprocessor = get_trained_model(20, CACHE_VERSION)
        st.session_state['main_model'] = model
        st.session_state['global_mae'] = mae
        st.session_state['training_preprocessor'] = preprocessor
        # Also set global for backward compatibility
        global TRAINING_PREPROCESSOR
        TRAINING_PREPROCESSOR = preprocessor
    return st.session_state['main_model'], st.session_state.get('global_mae', None)

data['DNF'] = data['DNF'].astype(int)

@st.cache_data
def get_dnf_diagnostic_probs(CACHE_VERSION):
    """Lazy-load DNF diagnostic logistic regression probabilities."""
    from sklearn.linear_model import LogisticRegression
    
    X_dnf, y_dnf = get_features_and_target_dnf(data)
    mask = y_dnf.notnull() & np.isfinite(y_dnf)
    X_dnf, y_dnf = X_dnf[mask], y_dnf[mask]
    if X_dnf.shape[0] == 0:
        return np.array([])
    else:
        preprocessor = get_preprocessor_dnf()
        X_dnf_prep = preprocessor.fit_transform(X_dnf)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_dnf_prep, y_dnf)
        return clf.predict_proba(X_dnf_prep)[:, 1]

@st.cache_data
def get_dnf_model(CACHE_VERSION, force_retrain=False):
    """Load or train the DNF prediction model."""
    if not force_retrain:
        pretrained = load_pretrained_model('dnf_model', CACHE_VERSION)
        if pretrained is not None:
            return pretrained['model']
    return train_and_evaluate_dnf_model(data, CACHE_VERSION)

@st.cache_data
def get_safetycar_model(CACHE_VERSION, force_retrain=False):
    """Load or train the safety car prediction model."""
    if not force_retrain:
        pretrained = load_pretrained_model('safetycar_model', CACHE_VERSION)
        if pretrained is not None:
            return pretrained['model']
    return train_and_evaluate_safetycar_model(safety_cars, CACHE_VERSION)

# Module-level execution guarded for headless imports
import os
if os.environ.get('STREAMLIT_SERVER_HEADLESS') != '1':
    X_sc, y_sc = get_features_and_target_safety_car(safety_cars)
    if X_sc.isnull().any().any():
        X_sc = X_sc.fillna(X_sc.mean(numeric_only=True))
else:
    # In headless mode, skip data loading
    X_sc, y_sc = None, None

def monte_carlo_feature_selection(
    X, y, model_class, n_trials=50, min_features=8, max_features=15, random_state=42, cv=5
):
    import random
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    results = []
    feature_names = X.columns.tolist()
    rng = random.Random(random_state)
    tested_subsets = set()
    for i in range(n_trials):
        subset = tuple(sorted(rng.sample(feature_names, k=rng.randint(min_features, max_features))))
        if subset in tested_subsets:
            continue
        tested_subsets.add(subset)
        X_subset = X[list(subset)].copy()
        # Convert object columns to category codes for XGBoost
        for col in X_subset.select_dtypes(include='object').columns:
            X_subset[col] = X_subset[col].astype('category').cat.codes
        for col in X_subset.select_dtypes(include='Int64').columns:
            X_subset[col] = X_subset[col].astype(float)
        X_subset = X_subset.fillna(X_subset.mean(numeric_only=True))
        # Remove rows with NaN or infinite values in y
        mask = y.notnull() & np.isfinite(y)
        X_subset_clean = X_subset[mask]
        y_clean = y[mask]
        # Cross-validation for MAE
        model = model_class()
        # Use the cleaned subset for all metrics
        mae_scores = cross_val_score(model, X_subset_clean, y_clean, cv=cv, scoring='neg_mean_absolute_error')
        mae = -mae_scores.mean()

        rmse_scores = cross_val_score(model, X_subset_clean, y_clean, cv=cv, scoring='neg_root_mean_squared_error')
        rmse = -rmse_scores.mean()

        r2_scores = cross_val_score(model, X_subset_clean, y_clean, cv=cv, scoring='r2')
        r2 = r2_scores.mean()
        results.append({'features': list(subset), 'mae': mae, 'rmse': rmse, 'r2': r2})
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

with tab1:
    st.header("Data Explorer")
    st.write("Filter and explore F1 race data from multiple perspectives.")
    
    if st.checkbox('Filter Results'):
        # Create a dictionary to store selected filters for multiple columns
        filters = {}
        filters_for_reset = {}

        # Iterate over the columns to display and create a filter for each
        st.sidebar.header("Select filters to apply:")
        for column in column_names:
            # derive a friendly name (if present) and a display label that
            # includes the raw field name in parentheses to help debugging
            column_friendly_name = column_rename_for_filter.get(column, column)
            # Historically we showed the raw field name for debugging:
            # display_label = f"{column_friendly_name} ({column})"
            # Only show the friendly label in the UI by default.
            display_label = column_friendly_name
            
            # Skip any columns explicitly in the exclusion list
            if column in exclusionList:
                continue

            # Detect if the column is boolean-like (actual boolean or 0/1 integer-like)
            is_bool_like = is_bool_dtype(data[column]) or (
                data[column].dropna().nunique() == 2 and set(data[column].dropna().unique()) <= {0, 1}
            )

            # Treat numeric columns as ranges except when they are boolean-like (e.g., DNF encoded 0/1)
            if is_numeric_dtype(data[column]) and not is_bool_like and (data[column].dtype in ('np.int64', 'np.float64', 'Int64', 'int64', 'Float64')):

                # Do not display if the column is in the exclusion list noted at the top of the file
                if column not in exclusionList:
                    min_val, max_val = int(data[column].min()), int(data[column].max())           
                                    
                    selected_range = st.sidebar.slider(
                        display_label,
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
                    display_label,
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

            elif is_bool_dtype(data[column]) or (
                # Treat 0/1 integer columns (like DNF) as boolean checkboxes
                data[column].dropna().nunique() == 2 and set(data[column].dropna().unique()) <= {0, 1}
            ):
                selected_value = st.sidebar.checkbox(
                    display_label,
                    value=False,
                    key=f"checkbox_filter_{column}"
                )
                if selected_value:
                    # For 0/1 columns we filter where value == 1
                    filters[column] = True

                # Store minimal reset info for checkbox filters
                filters_for_reset[column] = {
                    'key': f"checkbox_filter_{column}",
                    'column': column,
                    'dtype': str(data[column].dtype),
                    'selected_range': selected_value
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
                        display_label,
                        unique_values,
                        key=f"filter_{column}"
                )

                    filters_for_reset[column] = {
                        'key': f"filter_{column}",
                        'column': column,
                        'dtype': str(data[column].dtype),
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
        # Compute correlation matrix for selected numeric fields (used elsewhere)
        positionCorrelation = compute_safe_correlation(filtered_data, [
            'lastFPPositionNumber', 'resultsFinalPositionNumber', 'resultsStartingGridPositionNumber','grandPrixLaps', 'averagePracticePosition', 'DNF', 'resultsTop10', 'resultsTop5', 'resultsPodium', 'streetRace', 'trackRace',
            'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions', 'turns', 'positionsGained', 'q1End', 'q2End', 'q3Top10',  'driverBestStartingGridPosition', 'yearsActive',
            'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalPolePositions', 'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 'positionsGained',
            'avgLapPace', 'finishingTime'
        ])

        ## Rename Correlation Rows safely to avoid length mismatches
        # Map original dataframe column names to friendly labels and apply only
        # to the actual returned index (which may have had duplicates removed).
        friendly_map = {
            'lastFPPositionNumber': 'Last FP.',
            'resultsFinalPositionNumber': 'Final Pos.',
            'resultsStartingGridPositionNumber': 'Starting Grid Pos.',
            'grandPrixLaps': 'Laps',
            'averagePracticePosition': 'Avg Practice Pos.',
            'DNF': 'DNF',
            'resultsTop10': 'Top 10',
            'resultsTop5': 'Top 5',
            'resultsPodium': 'Podium',
            'streetRace': 'Street',
            'trackRace': 'Track',
            'constructorTotalRaceStarts': 'Constructor Race Starts',
            'constructorTotalRaceWins': 'Constructor Total Race Wins',
            'constructorTotalPolePositions': 'Constructor Pole Pos.',
            'turns': 'Turns',
            'positionsGained': 'Positions Gained',
            'q1End': 'Out at Q1',
            'q2End': 'Out at Q2',
            'q3Top10': 'Q3 Top 10',
            'driverBestStartingGridPosition': 'Best Starting Grid Pos.',
            'yearsActive': 'Years Active',
            'driverBestRaceResult': 'Best Result',
            'driverTotalChampionshipWins': 'Total Championship Wins',
            'driverTotalPolePositions': 'Total Pole Positions',
            'driverTotalRaceEntries': 'Race Entries',
            'driverTotalRaceStarts': 'Race Starts',
            'driverTotalRaceWins': 'Race Wins',
            'driverTotalRaceLaps': 'Race Laps',
            'driverTotalPodiums': 'Total Podiums',
            'avgLapPace': 'Avg. Lap Pace',
            'finishingTime': 'Finishing Time'
        }

        if not positionCorrelation.empty:
            actual_cols = list(positionCorrelation.index)
            new_labels = [friendly_map.get(c, c) for c in actual_cols]
            # Apply to both index and columns to keep matrix symmetric
            positionCorrelation.index = new_labels
            positionCorrelation.columns = new_labels

        # Create inner tabs so users can view the filtered data or the Data & Debug tools (including the leakage audit)
        data_tab, data_debug_tab = st.tabs(["Data", "Data & Debug"])

        with data_tab:
            st.dataframe(filtered_data, column_config=columns_to_display, column_order=['grandPrixYear', 'grandPrixName', 'streetRace', 'trackRace', 'constructorName', 'resultsDriverName', 'resultsPodium', 'resultsTop5',
                 'resultsTop10','resultsStartingGridPositionNumber','resultsFinalPositionNumber','positionsGained', 'DNF', 'resultsQualificationPositionNumber',
                   'q1End', 'q2End', 'q3Top10', 'averagePracticePosition',  'lastFPPositionNumber','numberOfStops', 'averageStopTime', 'totalStopTime',
                   'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalPolePositions', 'resultsReasonRetired',
                   'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 'positionsGained', 'avgLapTime', 'finishingTime'
                   ], hide_index=True, width=2400, height=600)

with tab2:
    st.header("Analytics & Visualizations")
    st.write("Comprehensive charts, regressions, and analysis of filtered data.")
    
    if 'filtered_data' in locals() and len(filtered_data) > 0:
        # Add visualizations for the filtered data
        st.subheader("Active Years v. Final Position")
        st.scatter_chart(filtered_data, x='resultsFinalPositionNumber', x_label='Final Position', y='yearsActive', y_label='Years Active', width="stretch")
        
        st.subheader("Positions Gained")
        st.line_chart(filtered_data, x='short_date', x_label='Date', y='positionsGained', y_label='Positions Gained', width="stretch")
        st.scatter_chart(filtered_data, x='short_date', x_label='Date', y='positionsGained', y_label='Positions Gained', width="stretch")
        
        st.subheader("Practice Position vs Final Position")
       
        st.scatter_chart(filtered_data, x='lastFPPositionNumber', x_label='Last FP Position', y='resultsFinalPositionNumber', y_label='Final Position', width="stretch")

        st.subheader("Starting Position vs Final Position")
        st.scatter_chart(filtered_data, x='resultsStartingGridPositionNumber', x_label='Starting Position', y='resultsFinalPositionNumber', y_label='Final Position', width="stretch")

        st.subheader("Average Practice Position vs Final Position")
        st.scatter_chart(filtered_data, x='averagePracticePosition', x_label='Average Practice Position', y='resultsFinalPositionNumber', y_label='Final Position', width="stretch")

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
            st.pyplot(fig, width="content")
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
            st.pyplot(fig, width="content")
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

        styled_correlation = positionCorrelation.style.map(highlight_correlation, subset=positionCorrelation.columns[1:])
        
        st.subheader("Correlation Matrix")
        st.caption("Correlation values range from -1 to 1, where -1 indicates a perfect negative correlation, 0 indicates no correlation, and 1 indicates a perfect positive correlation.")
        # Display the correlation matrix
        st.subheader("Feature Correlations with Final Position and Podium")
        # Select only relevant columns for review
        relevant_corr_cols = [
            'resultsFinalPositionNumber', 'resultsPodium', 'resultsTop5', 'resultsTop10',
            'resultsStartingGridPositionNumber', 'positionsGained', 'averagePracticePosition',
            'grandPrixLaps', 'lastFPPositionNumber', 'resultsQualificationPositionNumber',
            'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions',
            'turns', 'numberOfStops', 'driverBestStartingGridPosition', 'driverBestRaceResult',
            'driverTotalChampionshipWins', 'yearsActive', 'driverTotalRaceEntries', 'driverTotalRaceStarts',
            'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 'driverTotalPolePositions',
            'streetRace', 'trackRace', 'avgLapPace', 'finishingTime', 'DNF'
        ]
        # Filter and sort by absolute correlation with final position
        # Convert correlation matrix to a DataFrame with the row labels as a column
        corr_df = positionCorrelation.reset_index().rename(columns={'index': 'Feature'})
        # Only keep relevant columns that actually exist in this correlation matrix
        existing_corr_cols = [c for c in relevant_corr_cols if c in corr_df.columns]
        selected_cols = ['Feature'] + existing_corr_cols
        corr_df = corr_df[selected_cols].copy()
        # If 'resultsFinalPositionNumber' is present, sort by its absolute correlation
        if 'resultsFinalPositionNumber' in corr_df.columns:
            corr_df = corr_df.sort_values(by='resultsFinalPositionNumber', key=lambda x: abs(x), ascending=False)
        st.dataframe(
            corr_df,
            column_config=correlation_columns_to_display,
            hide_index=True,
            width='stretch',
            height=600
        )

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
        st.altair_chart(chart, width='stretch')

        constructor_performance = filtered_data.groupby(['grandPrixYear', 'constructorName']).agg(
        total_wins=('resultsFinalPositionNumber', lambda x: (x == 1).sum()),
        total_podiums=('resultsPodium', 'sum'),
        total_pole_positions=('constructorTotalPolePositions', 'sum')
        ).reset_index()
        
            
        st.subheader("Constructor Dominance Over the Years")
        st.bar_chart(constructor_performance, x='grandPrixYear', y=['total_wins', 'total_podiums'], color='constructorName', x_label='Year', y_label='Wins and Podiums', width="stretch")

        st.subheader("Impact of Starting Grid Position on Final Position")
        st.scatter_chart(filtered_data, x='resultsStartingGridPositionNumber', x_label='Starting Pos.', y_label='Final Pos.', y='resultsFinalPositionNumber', width="stretch")

        st.subheader("Pit Stop Analysis")
        st.scatter_chart(filtered_data, x='averageStopTime', x_label='Avg. Stop Time', y='resultsFinalPositionNumber', y_label='Final Pos.', width="stretch")

        driver_vs_constructor = filtered_data.groupby(['constructorName', 'resultsDriverName']).agg(
        positionsGained=('positionsGained', 'sum'),
        average_final_position=('resultsFinalPositionNumber', 'mean')
        ).reset_index()
        
        st.subheader("Driver vs Constructor Performance")
        driver_vs_constructor['average_final_position'] = driver_vs_constructor['average_final_position'].round(2)

        driver_vs_constructor = driver_vs_constructor.sort_values(by='average_final_position', ascending=True)
        st.dataframe(driver_vs_constructor, hide_index=True, column_config=driver_vs_constructor_columns_to_display, width=800,
        height=600,)

        # dnf_reasons = filtered_data[filtered_data['DNF']].groupby('resultsReasonRetired').size().reset_index(name='count')
        # dnf_reasons = filtered_data[filtered_data['DNF'].fillna(False)].groupby('resultsReasonRetired').size().reset_index(name='count')
        dnf_reasons = filtered_data[filtered_data['DNF'] == 1].groupby('resultsReasonRetired').size().reset_index(name='count')
        
        st.subheader("Reasons for DNFs")
        st.bar_chart(dnf_reasons, x='resultsReasonRetired', x_label='Reason', y='count', y_label='Count', width="stretch")

        # Count the number of entries (drivers) for each driver
        dnf_counts = (
        filtered_data[filtered_data['DNF']== 1]
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
        filtered_data[filtered_data['DNF']==1]
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

        height = get_dataframe_height(race_dnf_stats)
        st.dataframe(race_dnf_stats, hide_index=True, width=800,
        height=height, column_order=['grandPrixName', 'race_entry_count', 'dnf_count', 'dnf_pct'],
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
        filtered_data[filtered_data['DNF']==1]
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

        height = get_dataframe_height(constructor_dnf_stats)
        st.dataframe(constructor_dnf_stats, hide_index=True, width=800,
        height=height, column_order=['constructorName', 'constructor_entry_count', 'dnf_count', 'dnf_pct'],
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
        st.altair_chart(dnf_pie_chart, width='stretch')

        st.subheader("Track Characteristics and Performance")
        st.scatter_chart(filtered_data, x='turns', y='resultsFinalPositionNumber', width="stretch", x_label='Turns', y_label='Final Position')

        season_summary = filtered_data[filtered_data['grandPrixYear'] == current_year].groupby('resultsDriverName').agg(
        positions_gained =('positionsGained', 'sum'),
        total_podiums=('resultsPodium', 'sum')
        ).reset_index()
        
        st.subheader(f"{current_year} Season Summary")
        height = get_dataframe_height(season_summary)
        st.dataframe(season_summary, hide_index=True, column_config=season_summary_columns_to_display, width=800,
        height=height,)

        driver_consistency = filtered_data.groupby('resultsDriverName').agg(
        finishing_position_std=('resultsFinalPositionNumber', 'std')
        ).reset_index()
        
        st.subheader("Driver Consistency")
        st.caption("(Lower is Better)")
        st.bar_chart(driver_consistency, x='resultsDriverName', x_label='Driver', y_label='Standard Deviation - Finishing', y='finishing_position_std', width="stretch")
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
            model, mse, r2, mae, mean_err, evals_result, _ = train_and_evaluate_model(filtered_data)

            st.write(f"Mean Squared Error: {mse:.3f}")

            st.write(f"R^2 Score: {r2:.3f}")
            st.write(f"Mean Absolute Error: {mae:.2f}")
            st.write(f"Mean Error: {mean_err:.2f}")
                   
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            preprocessor = get_preprocessor_position(X)
            preprocessor.fit(X_train)  # Fit on training data
            X_test_prep = preprocessor.transform(X_test)

            # Predict based on model type
            if isinstance(model, xgb.Booster):  # XGBoost
                y_pred = model.predict(xgb.DMatrix(X_test_prep))
            else:  # LightGBM, CatBoost, sklearn models
                y_pred = model.predict(X_test_prep)

            # Create a DataFrame to display the features and predictions
            results_df = X_test.copy()
            results_df['Actual'] = y_test.values
            results_df['Predicted'] = y_pred
            results_df['Error'] = results_df['Actual'] - results_df['Predicted']

            # Ensure some common metadata columns are available for display (if present in the original filtered dataset)
            meta_cols = ['grandPrixName', 'constructorName', 'resultsDriverName']
            for col in meta_cols:
                if col not in results_df.columns and 'filtered_data' in globals() and col in filtered_data.columns:
                    try:
                        # Align by index from the split (train_test_split preserves the DataFrame index)
                        results_df[col] = filtered_data.loc[results_df.index, col]
                    except Exception:
                        # If alignment fails, fall back to adding a column of NaNs so later display logic can ignore it
                        results_df[col] = pd.NA

            # Select the top 3 actual finishers in each race (or just overall if not grouped by race)
            top3_actual = results_df.nsmallest(3, 'Actual')

            # Calculate MAE for the top 3
            mae_top3 = mean_absolute_error(top3_actual['Actual'], top3_actual['Predicted'])
            st.write(f"Mean Absolute Error (MAE) for Top 3 Podium Drivers: {mae_top3:.3f}")

            # Optionally, display the top 3 actual vs predicted
            st.subheader("Top 3 Podium Drivers: Actual vs Predicted")
            # Only request columns that actually exist to avoid KeyError
            display_cols = [c for c in ['grandPrixName', 'constructorName', 'resultsDriverName', 'Actual', 'Predicted', 'Error'] if c in top3_actual.columns]
            st.dataframe(top3_actual[display_cols], hide_index=True)

            # Display the first 30 rows
            st.subheader("First 30 Results with Accuracy")
            # Build a safe column order and column config based on what's actually present
            wanted_order = ['grandPrixName', 'constructorName', 'resultsDriverName', 'Actual', 'Predicted', 'Error']
            present_order = [c for c in wanted_order if c in results_df.columns]
            column_config = {}
            if 'grandPrixName' in results_df.columns:
                column_config['grandPrixName'] = st.column_config.TextColumn("Grand Prix")
            if 'constructorName' in results_df.columns:
                column_config['constructorName'] = st.column_config.TextColumn("Constructor")
            if 'resultsDriverName' in results_df.columns:
                column_config['resultsDriverName'] = st.column_config.TextColumn("Driver")
            if 'Actual' in results_df.columns:
                column_config['Actual'] = st.column_config.NumberColumn("Actual Pos.", format="%d")
            if 'Predicted' in results_df.columns:
                column_config['Predicted'] = st.column_config.NumberColumn("Predicted Pos.", format="%.3f")
            if 'Error' in results_df.columns:
                column_config['Error'] = st.column_config.NumberColumn("Error", format="%.3f")

            st.dataframe(results_df[present_order].head(30), hide_index=True, column_order=present_order, column_config=column_config)

            # Display feature importances
            st.subheader("Feature Importance")
            
            # Get feature names from your preprocessor
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

            # Get importances based on model type
            if hasattr(model, 'get_booster'):  # XGBoost (XGBRegressor)
                importances_dict = model.get_booster().get_score(importance_type='weight')
                importances = []
                for i, name in enumerate(feature_names):
                    importances.append(importances_dict.get(f'f{i}', 0))
            elif hasattr(model, 'feature_importances_'):  # LightGBM, CatBoost, sklearn models
                importances = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):  # CatBoost
                importances = model.get_feature_importance()
            else:  # Ensemble or other
                importances = [0] * len(feature_names)  # Default to zero

            # Create DataFrame for display
            feature_importances_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Percentage': np.array(importances) / (np.sum(importances) or 1) * 100
            }).sort_values(by='Importance', ascending=False)

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
    else:
        st.info("Please filter results in the Data Explorer tab first to view analytics.")

with tab3:
    st.header(f"{current_year} Season")
    st.write(f"Complete schedule and information for the {current_year} Formula 1 season.")
    
    # if st.checkbox(f"Show {current_year} Schedule", value=True):
    raceSchedule_display = raceSchedule[raceSchedule['year'] == current_year].copy()
    st.write(f"Total number of races: {len(raceSchedule_display)}")

    # Highlight the current week (next race) in a different color within the main schedule table
    today = datetime.datetime.today().date()
    raceSchedule_display['date_only'] = pd.to_datetime(raceSchedule_display['date']).dt.date
    next_race_date = raceSchedule_display[raceSchedule_display['date_only'] >= today]['date_only'].min()

    def highlight_current_week(row):
        color = 'background-color: #ffe599' if row['date_only'] == next_race_date else ''
        return [color] * len(row)

    # Apply styling directly to the main schedule table
    styled_schedule = raceSchedule_display.style.apply(highlight_current_week, axis=1)

    st.dataframe(
        styled_schedule,
        column_config=schedule_columns_to_display,
        hide_index=True,
        width=1000,
        height=900,
        column_order=['round', 'fullName', 'date', 'time', 'circuitType', 'courseLength', 'laps', 'turns', 'distance', 'totalRacesHeld']
    )

with tab4:
    st.header("Next Race")
    st.write("Details, predictions, and analysis for the upcoming race.")
    
    if st.checkbox("Show Next Race", value=True):
        st.subheader("Next Race:")
    
    # include the current date in the raceSchedule
    raceSchedule['date_only'] = pd.to_datetime(raceSchedule['date']).dt.date
    nextRace = raceSchedule[raceSchedule['date_only'] >= datetime.datetime.today().date()]

    # Create a copy of the slice to avoid the warning 
    nextRace = nextRace.sort_values(by=['date'], ascending = True).head(1)
    
    st.dataframe(nextRace, width=800, column_config=next_race_columns_to_display, hide_index=True, 
        column_order=['date', 'time', 'fullName', 'courseLength', 'turns', 'laps'])
    
    # Limit detailsOfNextRace by the grandPrixId of the next race
    if nextRace.empty:
        st.warning('No upcoming race found in the schedule.')
        next_race_id = None
        upcoming_race = pd.DataFrame()
        upcoming_race_id = []
    else:
        # safer scalar access
        next_race_id = nextRace['grandPrixId'].iat[0]
        upcoming_race = pd.merge(nextRace, raceSchedule, left_on='grandPrixId', right_on='grandPrixId', how='inner', suffixes=('_nextrace', '_races'))
        upcoming_race = upcoming_race.sort_values(by='date_nextrace', ascending = False).head(1)
        upcoming_race_id = upcoming_race['id_grandPrix_nextrace'].unique()

    st.subheader("Past Results:")
    if next_race_id is None:
        detailsOfNextRace = pd.DataFrame(columns=data.columns)
    else:
        detailsOfNextRace = data[data['grandPrixRaceId'] == next_race_id]

    # Sort detailsOfNextRace by grandPrixYear descending and resultsFinalPositionNumber ascending
    detailsOfNextRace = detailsOfNextRace.sort_values(by=['grandPrixYear', 'resultsFinalPositionNumber'], ascending=[False, True])

    st.write(f"Total number of results: {len(detailsOfNextRace)}")
    #detailsOfNextRace = detailsOfNextRace.drop_duplicates()
    detailsOfNextRace = detailsOfNextRace.drop_duplicates(subset=['resultsDriverName', 'grandPrixYear'])
    st.dataframe(detailsOfNextRace, column_config=columns_to_display, hide_index=True)
    
    # Safely pick a reference past race row if available
    if len(detailsOfNextRace) > 1:
        last_race = detailsOfNextRace.iloc[1]
    elif len(detailsOfNextRace) == 1:
        last_race = detailsOfNextRace.iloc[0]
    else:
        last_race = None

    latest_year = data['grandPrixYear'].max()
    active_driver_ids = data[data['grandPrixYear'] == latest_year]['resultsDriverId'].unique()
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
    # Compute means only for numeric columns (avoid assigning floats into boolean columns)
    numeric_means = input_data_next_race.select_dtypes(include='number').mean()
    # Fill missing values for numeric columns only using the computed means
    input_data_next_race = input_data_next_race.fillna(numeric_means)

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

    # Get MAE by position from Tab 5's session state
    position_mae_dict = st.session_state.get('position_mae_dict', {})

    # Narrow practices/qualifying to the relevant race.
    # Be defensive: `next_race_id` or `last_race` may be None or missing.
    if next_race_id is None:
        practices = practices.iloc[0:0]
        qualifying = qualifying.iloc[0:0]
    else:
        try:
            # If next race appears in practices dataset, use upcoming_race_id if available
            if 'raceId' in practices.columns and next_race_id in practices['raceId'].values:
                race_key = upcoming_race_id[0] if (isinstance(upcoming_race_id, (list, tuple)) and len(upcoming_race_id) > 0) else next_race_id
                practices = practices[practices['raceId'] == race_key]
                qualifying = qualifying[qualifying['raceId'] == race_key]
            else:
                # Fall back to last_race if available
                if last_race is not None and 'raceId_results' in last_race and pd.notna(last_race['raceId_results']):
                    lr = last_race['raceId_results']
                    practices = practices[practices['raceId'] == lr]
                    qualifying = qualifying[qualifying['raceId'] == lr]
                else:
                    practices = practices.iloc[0:0]
                    qualifying = qualifying.iloc[0:0]
        except Exception:
            practices = practices.iloc[0:0]
            qualifying = qualifying.iloc[0:0]

    # Choose which FP session to use. If `nextRace` is empty, keep practices empty.
    if not nextRace.empty:
        if nextRace.get('freePractice2Date', pd.Series()).isnull().all():
            practices = practices[practices['Session'] == 'FP1']
        else:
            practices = practices[practices['Session'] == 'FP2']

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
                # Filter out empty entries before combine_first to avoid FutureWarning
                mask = ~all_active_driver_inputs[latest_col].isnull()
                if mask.any():  # Only proceed if there are non-null values to combine
                    # Use fillna instead of combine_first to avoid the deprecated behavior
                    # Use infer_objects immediately after fillna to avoid pandas'
                    # silent downcasting FutureWarning (call on the result).
                    try:
                        tmp_series = all_active_driver_inputs.loc[mask, col].fillna(
                            all_active_driver_inputs.loc[mask, latest_col]
                        ).infer_objects(copy=False)
                    except Exception:
                        # Fall back to the original fillna result if infer_objects fails
                        tmp_series = all_active_driver_inputs.loc[mask, col].fillna(all_active_driver_inputs.loc[mask, latest_col])
                    all_active_driver_inputs.loc[mask, col] = tmp_series
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

    if X_predict.shape[0] == 0:
        st.error("No data available for prediction. This may be because there are no active drivers or no historical data for the upcoming race.")
        st.stop()
    
    # commented out on 9/17/2025 for early stopping
    # predicted_position = model.predict(X_predict)
    # Get preprocessor from session state (set by get_main_model)
    preprocessor = st.session_state.get('training_preprocessor')
    all_preprocessor_columns = []  # Initialize to prevent NameError in headless mode
    
    if preprocessor is None:
        # In headless mode or before model is loaded, skip predictions
        import os
        if os.environ.get('STREAMLIT_SERVER_HEADLESS') != '1':
            st.error("CRITICAL: Preprocessor not found in session state! The model's preprocessor was not loaded.")
            st.error("This will cause feature shape mismatch. Check that models include 'preprocessor' key in pickle artifact.")
            st.stop()
    else:
        # Only execute prediction code if preprocessor is loaded
        for name, _, cols in preprocessor.transformers:
            all_preprocessor_columns.extend(cols)
        missing_cols = [col for col in all_preprocessor_columns if col not in X_predict.columns]
        if missing_cols:
            st.error(f"These columns are missing from your prediction data and required by the training preprocessor: {missing_cols}")
            st.stop()

    # Fill any all-NaN columns expected by the preprocessor
    for col in all_preprocessor_columns:
        if col in X_predict.columns and X_predict[col].isnull().all():
            try:
                tmp_series = X_predict[col].fillna(0).infer_objects(copy=False)
            except Exception:
                tmp_series = X_predict[col].fillna(0)
            X_predict.loc[:, col] = tmp_series

    X_predict_prep = preprocessor.transform(X_predict)
    
    # Runtime diagnostics: when DEBUG enabled, emit model and feature info
    if DEBUG:
        try:
            debug_log("Model type", type(model))
            # XGBoost trained via sklearn wrapper
            try:
                booster = None
                if hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                elif isinstance(model, xgb.Booster):
                    booster = model
                if booster is not None and hasattr(booster, 'num_features'):
                    debug_log("XGBoost booster.num_features()", booster.num_features())
            except Exception as _e:
                debug_log("Could not read XGBoost booster features", str(_e))

            # sklearn-style attribute
            try:
                if hasattr(model, 'n_features_in_'):
                    debug_log('model.n_features_in_', getattr(model, 'n_features_in_', None))
            except Exception:
                pass

            debug_log('X_predict.shape', X_predict.shape)
            try:
                debug_log('X_predict_prep.shape', X_predict_prep.shape)
            except Exception:
                debug_log('X_predict_prep', type(X_predict_prep))

            # Feature names expected by preprocessor
            if TRAINING_PREPROCESSOR is not None:
                try:
                    feat_names = TRAINING_PREPROCESSOR.get_feature_names_out()
                except Exception:
                    feat_names = []
                    for name, _, cols in TRAINING_PREPROCESSOR.transformers:
                        feat_names.extend(cols)
                debug_log('training_preprocessor_feature_count', len(feat_names))
                debug_log('training_preprocessor_feature_sample', feat_names[:50])
            else:
                debug_log('TRAINING_PREPROCESSOR not set', None)
        except Exception as _ex:
            debug_log('Diagnostics error', str(_ex))

    # All sklearn-style models (XGBoost, LightGBM, CatBoost) accept numpy arrays
    # The preprocessed data X_predict_prep is already in the correct format (47 features)
    model, _ = get_main_model()
    predicted_position = model.predict(X_predict_prep)

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

    if X_predict_dnf.shape[0] == 0:
        st.warning("DNF prediction input is empty; skipping DNF probability predictions.")
        predicted_dnf_proba = np.array([])
    else:
        if X_predict_dnf.isnull().any().any():
            # st.warning("Imputing missing values in X_predict_dnf before prediction.")
            X_predict_dnf = X_predict_dnf.fillna(X_predict_dnf.mean(numeric_only=True))
        predicted_dnf_proba = get_dnf_model(CACHE_VERSION).predict_proba(X_predict_dnf)[:, 1]  # Probability of DNF=True

    
    # Holdout year evaluation for Safety Car Model
    train = safety_cars[safety_cars['grandPrixYear'] < current_year]
    test = safety_cars[safety_cars['grandPrixYear'] == current_year]
    X_train, y_train = get_features_and_target_safety_car(train)
    X_test, y_test = get_features_and_target_safety_car(test)

    holdout_model = train_and_evaluate_safetycar_model(train, CACHE_VERSION)
    # Now use holdout_model for predictions:
    # y_pred = holdout_model.predict_proba(X_test)[:, 1]
    if X_test.shape[0] == 0:
        st.warning("Safety Car holdout test set is empty; skipping predictions.")
        y_pred = np.array([])
    else:
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
        holdout_model = train_and_evaluate_safetycar_model(train, CACHE_VERSION)
        if X_test.shape[0] == 0:
            st.warning(f"Safety Car holdout evaluation for year {holdout_year} skipped: test set is empty.")
            y_pred = np.array([])
        else:
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

    # Populate synthetic row from `nextRace` when available; otherwise keep NaN
    if not nextRace.empty:
        # use .iat for scalar access
        if 'year' in nextRace.columns:
            synthetic_row['grandPrixYear'] = nextRace['year'].iat[0]
        if 'fullName' in nextRace.columns:
            synthetic_row['grandPrixName'] = nextRace['fullName'].iat[0]
    else:
        synthetic_row['grandPrixYear'] = np.nan
        synthetic_row['grandPrixName'] = np.nan
    
    # Fill with available info from nextRace, schedule, weather, etc.
    for col in ['circuitId', 'grandPrixLaps', 'turns', 'streetRace', 'trackRace']:
        if not nextRace.empty and col in nextRace.columns:
            synthetic_row[col] = nextRace[col].iat[0]

    # Fill weather features if available
    weather_row = weatherData[weatherData['grandPrixId'] == next_race_id]
    if not weather_row.empty:
        for col in ['average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation']:
            if col in weather_row.columns:
                # safe scalar access
                try:
                    synthetic_row[col] = weather_row[col].iat[0]
                except Exception:
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
    if X_predict_safetycar.shape[0] == 0:
        st.warning("Synthetic safety-car feature row is empty; skipping safety car probability for next race.")
        safety_car_proba = np.nan
    else:
        if X_predict_safetycar.isnull().any().any():
            X_predict_safetycar = X_predict_safetycar.fillna(X_predict_safetycar.mean(numeric_only=True))
        safety_car_proba = get_safetycar_model(CACHE_VERSION).predict_proba(X_predict_safetycar)[:, 1][0]
    
    # Add both to your DataFrame
    all_active_driver_inputs['PredictedFinalPosition'] = predicted_position
    all_active_driver_inputs['PredictedDNFProbability'] = predicted_dnf_proba
    all_active_driver_inputs['PredictedDNFProbabilityPercentage'] = (all_active_driver_inputs['PredictedDNFProbability'] * 100).round(3)
    _, global_mae = get_main_model()
    all_active_driver_inputs['PredictedFinalPosition_Low'] = (all_active_driver_inputs['PredictedFinalPosition'] - global_mae).astype(float)
    all_active_driver_inputs['PredictedFinalPosition_High'] = (all_active_driver_inputs['PredictedFinalPosition'] + global_mae).astype(float)

    # Get latest DNF stats for each active driver from the main dataset
    latest_dnf_stats = (
        data.sort_values('grandPrixYear')
            .groupby('resultsDriverId')
            .tail(1)[['resultsDriverId', 'driverDNFCount', 'driverDNFAvg']]
    )

    # Merge the latest stats
    all_active_driver_inputs = pd.merge(
        all_active_driver_inputs,
        latest_dnf_stats,
        on='resultsDriverId',
        how='left',
        suffixes=('', '_latest')
    )

    # Use latest stats if available, otherwise fill with 0
    all_active_driver_inputs['driverDNFCount'] = (
        all_active_driver_inputs['driverDNFCount_latest']
        .fillna(all_active_driver_inputs['driverDNFCount'])
        .fillna(0)
        .astype(int)
    )

    all_active_driver_inputs['driverDNFAvg'] = (
        all_active_driver_inputs['driverDNFAvg_latest']
        .fillna(all_active_driver_inputs['driverDNFAvg'])
        .fillna(0.0)
        .astype(float)
    )

    # Clean up temporary columns
    all_active_driver_inputs = all_active_driver_inputs.drop(columns=['driverDNFCount_latest', 'driverDNFAvg_latest'], errors='ignore')

    # Calculate percentage
    all_active_driver_inputs['driverDNFPercentage'] = (all_active_driver_inputs['driverDNFAvg'] * 100).round(3)

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
    all_active_driver_inputs['Historical MAE by Rank'] = all_active_driver_inputs['Rank'].map(position_mae_dict)
    all_active_driver_inputs = all_active_driver_inputs.set_index('Rank')

    # Fix the column data for display after the merge
    all_active_driver_inputs.drop(columns=['constructorName', 'constructorName_y'], inplace=True, errors='ignore')
    all_active_driver_inputs = all_active_driver_inputs.rename(columns={'constructorName_x': 'constructorName'})

    # Calculate MAE by individual positions for mapping to predicted positions
    X_mae, y_mae = get_features_and_target(data)
    X_train_mae, X_test_mae, y_train_mae, y_test_mae = train_test_split(X_mae, y_mae, test_size=0.2, random_state=42)
    preprocessor_mae = get_preprocessor_position(X_mae)
    preprocessor_mae.fit(X_train_mae)
    X_test_prep_mae = preprocessor_mae.transform(X_test_mae)
    
    # Predict based on model type
    if isinstance(model, xgb.Booster):  # XGBoost
        y_pred_mae = model.predict(xgb.DMatrix(X_test_prep_mae))
    else:  # LightGBM, CatBoost, sklearn models
        y_pred_mae = model.predict(X_test_prep_mae)
    
    results_df_analysis_mae = pd.DataFrame({
        'Actual': y_test_mae.values,
        'Predicted': y_pred_mae
    })
    
    individual_mae = []
    for pos in range(1, 21):
        pos_data = results_df_analysis_mae[results_df_analysis_mae['Actual'] == pos]
        if len(pos_data) > 0:
            mae_pos = mean_absolute_error(pos_data['Actual'], pos_data['Predicted'])
            individual_mae.append({
                'Position': pos,
                'MAE': mae_pos,
                'Sample Size': len(pos_data)
            })
    
    individual_mae_df = pd.DataFrame(individual_mae)
    
    # Create mapping from position to MAE
    mae_by_position = dict(zip(individual_mae_df['Position'], individual_mae_df['MAE']))
    
    # Add MAE for predicted position to the dataframe
    _, global_mae = get_main_model()
    all_active_driver_inputs['PredictedPositionMAE'] = (
        all_active_driver_inputs.index
        .map(mae_by_position)
        .fillna(global_mae)
    )

    all_active_driver_inputs['PredictedPositionMAE_Low'] = all_active_driver_inputs['PredictedFinalPosition'] - all_active_driver_inputs['PredictedPositionMAE']
    all_active_driver_inputs['PredictedPositionMAE_High'] = all_active_driver_inputs['PredictedFinalPosition'] + all_active_driver_inputs['PredictedPositionMAE']

    # st.write(all_active_driver_inputs.columns.tolist())
    st.subheader("Predictive Results for Active Drivers")

    st.write(f"MAE for Position Predictions: {global_mae:.3f}")
    height = get_dataframe_height(all_active_driver_inputs)
    st.dataframe(all_active_driver_inputs, hide_index=False, column_config=predicted_position_columns_to_display, width=1200, height=height, 
    column_order=['constructorName', 'resultsDriverName', 'PredictedFinalPosition', 'PredictedFinalPositionStd', 'PredictedFinalPosition_Low', 'PredictedFinalPosition_High', 'PredictedPositionMAE', 'PredictedPositionMAE_Low', 'PredictedPositionMAE_High'])

    st.subheader("Predictive DNF")

    st.write("Logistic Regression DNF Probabilities:")
    probs = get_dnf_diagnostic_probs(CACHE_VERSION)
    st.write("Min:", probs.min(), "Max:", probs.max(), "Mean:", probs.mean())

    all_active_driver_inputs.sort_values(by='PredictedDNFProbabilityPercentage', ascending=False, inplace=True)
    height = get_dataframe_height(all_active_driver_inputs)
    st.dataframe(all_active_driver_inputs, hide_index=False, column_config=predicted_dnf_position_columns_to_display, width=800, height=height, 
    column_order=['constructorName', 'resultsDriverName', 'driverDNFCount',  'driverDNFPercentage', 'PredictedDNFProbabilityPercentage', 'PredictedDNFProbabilityStd'], )  

    st.subheader("Predicted Safety Car")

    # Ensure race_level is a copy to avoid SettingWithCopyWarning
    race_level = race_level.copy()  # Add this line before assignment if not already a copy
    
    X_sc, y_sc = get_features_and_target_safety_car(safety_cars)
    if X_sc.shape[0] == 0:
        st.warning("No safety-car historical features available; skipping bulk safety car predictions.")
        safety_cars['PredictedSafetyCarProbability'] = np.nan
    else:
        if X_sc.isnull().any().any():
            X_sc = X_sc.fillna(X_sc.mean(numeric_only=True))
        safety_cars['PredictedSafetyCarProbability'] = get_safetycar_model(CACHE_VERSION).predict_proba(X_sc)[:, 1]
    safety_cars['PredictedSafetyCarProbabilityPercentage'] = (safety_cars['PredictedSafetyCarProbability'] * 100).round(3)

    historical_display = safety_cars[['grandPrixName', 'grandPrixYear', 'PredictedSafetyCarProbabilityPercentage']].copy()
    historical_display['Type'] = 'Historical'

    synthetic_df['PredictedSafetyCarProbability'] = safety_car_proba
    synthetic_df['PredictedSafetyCarProbabilityPercentage'] = (synthetic_df['PredictedSafetyCarProbability'] * 100).round(3)
    synthetic_df['Type'] = 'Next Race'

    # Only show historical and synthetic predictions for the current Grand Prix
    if not synthetic_df.empty:
        current_gp_name = synthetic_df['grandPrixName'].iat[0] if 'grandPrixName' in synthetic_df.columns else None
        current_gp_year = synthetic_df['grandPrixYear'].iat[0] if 'grandPrixYear' in synthetic_df.columns else None
    else:
        current_gp_name = None
        current_gp_year = None

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

    height = get_dataframe_height(display_df)
    st.dataframe(
        display_df[['grandPrixName', 'grandPrixYear', 'PredictedSafetyCarProbabilityPercentage', 'Type']].sort_values(by=['grandPrixYear'], ascending=[False]),
        hide_index=True,
        width=800,
        height=height,
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
    if not nextRace.empty:
        predicted_results['grandPrixName'] = nextRace['fullName'].iat[0] if 'fullName' in nextRace.columns else pd.NA
        predicted_results['grandPrixYear'] = nextRace['year'].iat[0] if 'year' in nextRace.columns else pd.NA
        year_for_fname = str(nextRace['year'].iat[0]) if 'year' in nextRace.columns and not nextRace['year'].isnull().all() else 'unknown'
    else:
        predicted_results['grandPrixName'] = pd.NA
        predicted_results['grandPrixYear'] = pd.NA
        year_for_fname = 'unknown'

    # Save to CSV for later comparison; guard filename construction
    fname = path.join(DATA_DIR, f"predictions_{next_race_id if next_race_id is not None else 'unknown'}_{year_for_fname}.csv")
    try:
        predicted_results.to_csv(fname, index=False)
    except Exception as _e:
        print(f"Could not write predictions file {fname}: {_e}")

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

    next_race_name = nextRace['fullName'].iat[0] if (not nextRace.empty and 'fullName' in nextRace.columns) else 'Unknown Grand Prix'
    st.subheader(f"Flags and Safety Cars from {next_race_name}:")
    st.caption("Race messages, including flags, are only available going back to 2018.")
    # race_control_messages_grouped_with_dnf.csv
    raceMessagesOfNextRace = race_messages[race_messages['grandPrixId'] == next_race_id]
    raceMessagesOfNextRace = raceMessagesOfNextRace.sort_values(by='Year', ascending = False)

    st.write(f"Total number of results: {len(raceMessagesOfNextRace)}")
    st.dataframe(raceMessagesOfNextRace, hide_index=True, width=800,column_config=flags_safety_cars_columns_to_display, 
                 column_order=['Year', 'Round', 'SafetyCarStatus', 'redFlag', 'yellowFlag', 'doubleYellowFlag', 'dnf_count'])

    st.subheader(f"Driver Performance in {next_race_name}:")
    st.write(f"Total number of results: {len(individual_race_grouped)}")
    
    # Display the grouped data without index
    st.dataframe(individual_race_grouped, hide_index=True, width=800, height=600, column_config=individual_race_grouped_columns_to_display)

    st.subheader(f"Constructor Performance in {next_race_name}:")
    height = get_dataframe_height(individual_race_grouped_constructor)
    st.dataframe(individual_race_grouped_constructor, hide_index=True, width=800, height=height, column_config=individual_race_grouped_columns_to_display)

     # Load raw pit stop data (individual stops)
    pitstops = pd.read_json(path.join(DATA_DIR, 'f1db-races-pit-stops.json'))
    pitstops = pitstops[pitstops['year'] >= 2018]  # Filter for recent years if needed
    # Filter for all prior races at the same Grand Prix as the next race
    # First, map raceId to grandPrixId using raceSchedule
    
    pitstops = pitstops.merge(
        raceSchedule[['id_grandPrix', 'grandPrixId', 'year', 'round']],
        left_on='raceId',
        right_on='id_grandPrix',
        how='left'
    )
    prior_gp_pitstops = pitstops[pitstops['grandPrixId'] == next_race_id]

    # st.write("Data columns:", data.columns.tolist())

    if not prior_gp_pitstops.empty:
        # Find the fastest individual pit stop per constructor per race
        fastest_pitstops = (
            prior_gp_pitstops.loc[

                prior_gp_pitstops.groupby(['raceId', 'constructorId'])['timeMillis'].idxmin()
            ]
            .sort_values(['raceId', 'constructorId', 'timeMillis'])
        )

        # Optionally, merge with constructor names if needed
        if 'constructorName' in data.columns and 'constructorId' in fastest_pitstops.columns:
            constructor_names = data[['constructorId_results', 'constructorName']].drop_duplicates()
            fastest_pitstops = fastest_pitstops.merge(constructor_names, left_on='constructorId', right_on='constructorId_results', how='left')

        # Convert milliseconds to seconds for display
        fastest_pitstops['pitStopSeconds'] = (fastest_pitstops['timeMillis'] / 1000).round(3)

        if 'year_x' in fastest_pitstops.columns and 'year' not in fastest_pitstops.columns:
            fastest_pitstops = fastest_pitstops.rename(columns={'year_x': 'year'})
        elif 'year_y' in fastest_pitstops.columns:
            fastest_pitstops = fastest_pitstops.rename(columns={'year_y': 'year'})

        if 'round_x' in fastest_pitstops.columns and 'round' not in fastest_pitstops.columns:
            fastest_pitstops = fastest_pitstops.rename(columns={'round_x': 'round'})
        elif 'round_y' in fastest_pitstops.columns:
            fastest_pitstops = fastest_pitstops.rename(columns={'round_y': 'round'})

        # Only assign if there is at least one value
        pit_lane_vals = data[data['grandPrixRaceId'] == next_race_id]['pit_lane_time_constant']
        if not pit_lane_vals.empty:
            fastest_pitstops['pit_lane_time_constant'] = pit_lane_vals.iloc[0]
        else:
            fastest_pitstops['pit_lane_time_constant'] = np.nan

        fastest_pitstops['pit_time_stationary'] = (fastest_pitstops['pitStopSeconds'] - fastest_pitstops['pit_lane_time_constant']).round(3)

        st.subheader("Fastest Individual Pit Stop per Constructor")
        st.write(f"Total number of fastest pit stops: {len(fastest_pitstops)}")
        # Safely display pit lane time constant
        pit_lane_const = "N/A"
        try:
            if len(fastest_pitstops) > 0:
                val = fastest_pitstops['pit_lane_time_constant'].dropna()
                if not val.empty:
                    pit_lane_const = val.iat[0]
        except Exception:
            pit_lane_const = "N/A"
        st.write(f"Pit Time Constant:", pit_lane_const)
        fastest_pitstops = fastest_pitstops.sort_values(by=['year', 'pitStopSeconds'], ascending=[False, True])
        height = get_dataframe_height(fastest_pitstops)
        st.dataframe(
            fastest_pitstops[['year', 'round', 'constructorName', 'lap', 'pitStopSeconds', 'pit_time_stationary']],
            hide_index=True,
            width=800,
            height=height,
            column_config={
                
                'year': st.column_config.NumberColumn("Year"),
                'round': st.column_config.NumberColumn("Round"),
                'constructorName': st.column_config.TextColumn("Constructor"),
                'lap': st.column_config.NumberColumn("Lap"),
                'pitStopSeconds': st.column_config.NumberColumn("Pit Stop (s)", format="%.3f"),
                'pit_time_stationary': st.column_config.NumberColumn("Pit Time Stationary (s)", format="%.3f"),
            }
        )
    else:
        st.info("No individual pit stop data available for prior races at this Grand Prix.")


    weather_with_grandprix = weatherData[weatherData['grandPrixId'] == next_race_id]
    
    weather_name = weather_with_grandprix['fullName'].iat[0] if (not weather_with_grandprix.empty and 'fullName' in weather_with_grandprix.columns) else 'Unknown Grand Prix'
    st.subheader(f"Weather Data for {weather_name}:")
    st.write(f"Total number of weather records: {len(weather_with_grandprix)}")

    weather_with_grandprix = weather_with_grandprix.sort_values(by='short_date', ascending = False)
    st.dataframe(weather_with_grandprix, width=800, column_config=weather_columns_to_display, hide_index=True)
    

with tab5:
    # Force immediate render to test if tab executes
    st.write("Tab 5 START")
    st.header("Predictive Models & Advanced Options")
    st.write("Advanced machine learning models, hyperparameter tuning, and feature selection tools.")
    
    # Model type selection
    model_type = st.selectbox(
        "Select Model Type", 
        ["XGBoost", "LightGBM", "CatBoost", "Ensemble (XGBoost + LightGBM + CatBoost)"],
        index=0,
        help="Choose the machine learning model to use for predictions"
    )
    
    # Model information expander
    with st.expander("ℹ️ Model Information & Recommendations", expanded=False):
        st.markdown("""
        ### Model Descriptions & Use Cases
        
        **🏆 XGBoost (Recommended Default)**
        - **Strengths**: Excellent performance, handles missing data, built-in feature importance, widely used in competitions
        - **Best for**: General-purpose predictions, when you want reliable and interpretable results
        - **Training speed**: Fast
        - **Memory usage**: Moderate
        - **Current MAE**: ~1.94 (baseline)
        
        **🚀 LightGBM**
        - **Strengths**: Very fast training, handles large datasets well, good for categorical features
        - **Best for**: When training speed is critical or working with large datasets
        - **Training speed**: Very fast
        - **Memory usage**: Low
        - **Note**: May be less interpretable than XGBoost
        
        **🐱 CatBoost**
        - **Strengths**: Excellent with categorical data, robust to overfitting, handles missing values automatically
        - **Best for**: Datasets with many categorical features or when robustness is important
        - **Training speed**: Moderate
        - **Memory usage**: Moderate
        - **Note**: Slower training but often more stable predictions
        
        **🎯 Ensemble (XGBoost + LightGBM + CatBoost)**
        - **Strengths**: Combines strengths of all three models, often better accuracy through diversity
        - **Best for**: Maximum prediction accuracy, when computational resources allow
        - **Training speed**: Slowest (trains 3 models + meta-learner)
        - **Memory usage**: High
        - **Note**: Recommended for final predictions or when comparing against single models
        
        ### Performance Expectations
        - **Single models**: Fast training (seconds), good accuracy
        - **Ensemble**: Slower training (minutes), potentially better accuracy
        - **All models** use early stopping to prevent overfitting
        - **Sample weighting** favors podium positions for better top-10 accuracy
        """)
    
    # Early stopping rounds - always visible at top
    early_stopping_rounds = st.number_input(
        "Early stopping rounds", min_value=1, max_value=100, value=20, step=1, 
        help="Number of rounds with no improvement to stop training"
    )

    # Store for use in Tab 4
    st.session_state['early_stopping_rounds'] = early_stopping_rounds

    # Train model once at the top level for reuse (lazy loading with cache)
    try:
        model, mse, r2, mae, mean_err, evals_result, preprocessor = get_trained_model(early_stopping_rounds, CACHE_VERSION)
        
        # Store preprocessor in session state
        st.session_state['training_preprocessor'] = preprocessor
    except Exception as e:
        st.error(f"CRITICAL ERROR in get_trained_model: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()
    
    # Single expander with 6 tabs inside
    with st.expander("🔧 Advanced Options", expanded=True):
        tab_perf, tab_feat, tab_select, tab_position, tab_hyper, tab_hist, tab_debug = st.tabs([
            "📊 Model Performance",
            "🔍 Feature Analysis", 
            "🎯 Feature Selection",
            "🏎️ Position-Specific Analysis",
            "⚙️ Hyperparameters",
            "📈 Historical Validation",
            "🛠️ Debug & Experiments"
        ])
        
        with tab_perf:
            st.subheader("Predictive Data Model Metrics")
            
            st.write(f"Mean Squared Error: {mse:.3f}")
            st.write(f"R^2 Score: {r2:.3f}")
            st.write(f"Mean Absolute Error: {mae:.2f}")
            st.write(f"Mean Error: {mean_err:.2f}")
            
            # Display boosting rounds used (model-specific)
            if hasattr(model, 'best_iteration'):
                st.write(f"Boosting rounds used: {model.best_iteration + 1}")
            elif hasattr(model, 'best_iteration_'):
                st.write(f"Boosting rounds used: {model.best_iteration_}")
            elif hasattr(model, 'get_best_iteration'):
                st.write(f"Boosting rounds used: {model.get_best_iteration()}")
            else:
                st.write("Model type: Ensemble or other (no boosting rounds info)")


            # Combine predictions and actuals for comparison
            X, y = get_features_and_target(data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            preprocessor = get_preprocessor_position(X)
            preprocessor.fit(X_train)
            X_test_prep = preprocessor.transform(X_test)
            
            # Predict based on actual model type
            if isinstance(model, xgb.Booster):
                y_pred = model.predict(xgb.DMatrix(X_test_prep))
            else:
                y_pred = model.predict(X_test_prep)


            results_df = X_test.copy()
            results_df['Actual'] = y_test.values
            results_df['Predicted'] = y_pred
            results_df['Error'] = results_df['Actual'] - results_df['Predicted']

            # Position-specific MAE analysis
            results_df_analysis = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred
            })

            podium_actual = results_df_analysis[results_df_analysis['Actual'] <= 3]
            points_actual = results_df_analysis[results_df_analysis['Actual'] <= 10]
            winners_actual = results_df_analysis[results_df_analysis['Actual'] == 1]
            bottom_10_actual = results_df_analysis[results_df_analysis['Actual'] >= 11]
            
            if len(podium_actual) > 0:
                podium_mae = mean_absolute_error(podium_actual['Actual'], podium_actual['Predicted'])
                st.write(f"MAE for Podium Finishers (1-3): {podium_mae:.3f}")
                
            if len(winners_actual) > 0:
                winner_mae = mean_absolute_error(winners_actual['Actual'], winners_actual['Predicted'])
                st.write(f"MAE for Race Winners: {winner_mae:.3f}")

            if len(points_actual) > 0:
                points_mae = mean_absolute_error(points_actual['Actual'], points_actual['Predicted'])
                st.write(f"MAE for Points Positions (1-10): {points_mae:.3f}")

            if len(bottom_10_actual) > 0:
                bottom_10_mae = mean_absolute_error(bottom_10_actual['Actual'], bottom_10_actual['Predicted'])
                st.write(f"MAE for Bottom 10 Positions (11-20): {bottom_10_mae:.3f}")

            # Driver error stats
            results_df['Error'] = results_df['Actual'] - results_df['Predicted']
            results_df['AbsError'] = results_df['Error'].abs()
            results_df['SquaredError'] = results_df['Error'] ** 2

            driver_error_stats = results_df.groupby('resultsDriverName').agg(
                MeanError=('Error', 'mean'),
                MeanAbsoluteError=('AbsError', 'mean'),
                RMSE=('SquaredError', lambda x: np.sqrt(np.mean(x))),
                MedianAbsoluteError=('AbsError', 'median'),
                MaxError=('Error', 'max'),
                MinError=('Error', 'min'),
                Count=('Error', 'count')
            ).reset_index()

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

            st.subheader("Predictive Results with Features")
            st.dataframe(results_df, hide_index=True, width='stretch')

            # Feature importances
            st.subheader("Feature Importances")
            feature_names = get_features_and_target(data)[0].columns.tolist()
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

            # Get importances based on model type
            if hasattr(model, 'get_booster'):  # XGBoost (XGBRegressor)
                importances_dict = model.get_booster().get_score(importance_type='weight')
                importances = []
                for i, name in enumerate(feature_names):
                    importances.append(importances_dict.get(f'f{i}', 0))
            elif hasattr(model, 'feature_importances_'):  # LightGBM, CatBoost, sklearn models
                importances = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):  # CatBoost
                importances = model.get_feature_importance()
            else:  # Ensemble or other
                importances = [0] * len(feature_names)  # Default to zero

            feature_importances_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Percentage': np.array(importances) / (np.sum(importances) or 1) * 100
            }).sort_values(by='Importance', ascending=False)

            # Display boosting rounds used (conditional on model type)
            if hasattr(model, 'best_iteration_'):  # LightGBM
                st.write(f"Boosting rounds used: {model.best_iteration_}")
            elif hasattr(model, 'best_iteration'):  # XGBoost
                st.write(f"Boosting rounds used: {model.best_iteration + 1}")
            elif hasattr(model, 'get_best_iteration'):  # CatBoost
                st.write(f"Boosting rounds used: {model.get_best_iteration()}")
            else:  # Ensemble or other
                st.write("Boosting rounds information not available for this model type")

            st.dataframe(feature_importances_df, hide_index=True, width=800)
            
            # MAE by Position Groups
            st.subheader("MAE by Position Groups")
            st.info("📊 This analysis uses a 20% test set. Some position ranges may not have data in the test set due to random sampling. This is normal and doesn't affect the overall model performance.")
            
            # Define position groups
            mid_field_actual = results_df_analysis[(results_df_analysis['Actual'] >= 11) & (results_df_analysis['Actual'] <= 15)]
            back_actual = results_df_analysis[results_df_analysis['Actual'] >= 16]
            
            position_groups = [
                ("Winner (P1)", winners_actual),
                ("Top 3 (P1-3)", podium_actual),
                ("Top 10 (P1-10)", points_actual),
                ("Mid-field (P11-15)", mid_field_actual),
                ("Back (P16-20)", back_actual),
                ("Bottom 10 (P11-20)", bottom_10_actual)
            ]
            
            mae_data = []
            for group_name, group_data in position_groups:
                if len(group_data) > 0:
                    mae = mean_absolute_error(group_data['Actual'], group_data['Predicted'])
                    mae_data.append({
                        'Position Group': group_name,
                        'MAE': mae,
                        'Sample Size': len(group_data)
                    })
            
            mae_df = pd.DataFrame(mae_data)
            st.dataframe(mae_df, hide_index=True, width=600)
            st.bar_chart(mae_df.set_index('Position Group')['MAE'], width='stretch')
            
            # Individual positions MAE
            st.subheader("MAE by Individual Positions")
            individual_mae = []
            for pos in range(1, 21):
                pos_data = results_df_analysis[results_df_analysis['Actual'] == pos]
                if len(pos_data) > 0:
                    mae = mean_absolute_error(pos_data['Actual'], pos_data['Predicted'])
                    individual_mae.append({
                        'Position': pos,
                        'MAE': mae,
                        'Sample Size': len(pos_data)
                    })
            
            individual_mae_df = pd.DataFrame(individual_mae)
            st.dataframe(individual_mae_df, hide_index=True, width=600, height=750)
            st.line_chart(individual_mae_df.set_index('Position')['MAE'], width='stretch')

            # Store for use in Tab 4
            st.session_state['position_mae_dict'] = dict(zip(individual_mae_df['Position'], individual_mae_df['MAE']))

            # Position group summary
            st.subheader("Position Group Summary")
            summary_data = []
            for group_name, group_data in position_groups:
                if len(group_data) > 0:
                    mae = mean_absolute_error(group_data['Actual'], group_data['Predicted'])
                    avg_error = (group_data['Predicted'] - group_data['Actual']).mean()
                    median_error = (group_data['Predicted'] - group_data['Actual']).median()
                    summary_data.append({
                        'Position Group': group_name,
                        'Sample Size': len(group_data),
                        'MAE': mae,
                        'Average Error': avg_error,
                        'Median Error': median_error
                    })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, hide_index=True, width=1000)
            
            # Error Distribution
            st.subheader("Prediction Error Distribution by Position Groups")
            results_df_analysis['AbsError'] = abs(results_df_analysis['Actual'] - results_df_analysis['Predicted'])
            results_df_analysis['Position_Group'] = pd.cut(
                results_df_analysis['Actual'], 
                bins=[0, 1, 3, 10, 15, 20], 
                labels=['Winner', 'Podium', 'Points', 'Mid-field', 'Back'],
                include_lowest=True
            )
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            position_groups_cat = results_df_analysis['Position_Group'].cat.categories
            error_data = [results_df_analysis[results_df_analysis['Position_Group'] == group]['AbsError'].values 
                        for group in position_groups_cat]
            ax.boxplot(error_data, tick_labels=position_groups_cat)
            ax.set_ylabel('Absolute Error')
            ax.set_xlabel('Position Group')
            ax.set_title('Prediction Error Distribution by Position Group')
            st.pyplot(fig, width=1000)
        
        with tab_feat:
            st.subheader("Feature Analysis")
            
            # Permutation Importance
            st.write("### Permutation Importance (Feature Impact on Model Error)")
            from sklearn.inspection import permutation_importance

            X_feat, y_feat = get_features_and_target(data)
            mask = y_feat.notnull() & np.isfinite(y_feat)
            X_feat, y_feat = X_feat[mask], y_feat[mask]
            preprocessor_feat = get_preprocessor_position(X_feat)
            X_prep_feat = preprocessor_feat.fit_transform(X_feat)
            model_feat = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=42)
            model_feat.fit(X_prep_feat, y_feat)

            result = permutation_importance(model_feat, X_prep_feat, y_feat, n_repeats=10, random_state=42)
            importances = result.importances_mean
            feature_names_perm = preprocessor_feat.get_feature_names_out()
            feature_names_perm = [name.replace('num__', '').replace('cat__', '') for name in feature_names_perm]

            perm_df = pd.DataFrame({
                'Feature': feature_names_perm,
                'Permutation Importance': importances
            }).sort_values(by='Permutation Importance', ascending=True)

            st.write("Features with lowest permutation importance (least helpful):")
            st.dataframe(perm_df.head(100), hide_index=True, width=800)
            st.write("Features with highest permutation importance (most helpful):")
            st.dataframe(perm_df.tail(100).sort_values(by='Permutation Importance', ascending=False), hide_index=True, width=800)

            # High-Cardinality Features
            st.write("### High-Cardinality Features (Potential Overfitting Risk)")
            X_card, _ = get_features_and_target(data)
            cardinality = X_card.nunique().sort_values(ascending=False)
            cardinality_df = pd.DataFrame({
                'Feature': cardinality.index,
                'Unique Values': cardinality.values
            })
            cardinality_df['Risk'] = np.where(cardinality_df['Unique Values'] > 50, 'High', 'Low')
            st.write("Features with high cardinality (many unique values) are more likely to cause overfitting, especially if they are IDs or post-event info.")
            st.dataframe(cardinality_df, hide_index=True, width=800)

            # Safety Car Data Importances
            st.write("### Safety Car Feature Importance")
            safetycar_model_loaded = get_safetycar_model(CACHE_VERSION)
            preprocessor_sc = safetycar_model_loaded.named_steps['preprocessor']
            feature_names_sc = preprocessor_sc.get_feature_names_out()
            feature_names_sc = [name.replace('num__', '').replace('cat__', '') for name in feature_names_sc]
            importances_sc = safetycar_model_loaded.named_steps['classifier'].coef_[0]

            odds_ratios = np.exp(importances_sc)
            prob_change = (1 / (1 + np.exp(-importances_sc))) - 0.5

            df_sc = pd.DataFrame({
                'Feature': feature_names_sc,
                'Coefficient': importances_sc,
                'Odds Ratio': odds_ratios,
                'Prob Change (per unit)': prob_change
            }).sort_values('Coefficient', key=np.abs, ascending=False, ignore_index=True)

            st.dataframe(df_sc, width=1000, hide_index=True)

            # Correlations
            st.write("### Correlation Matrix")
            # Get the original DataFrame from the Styler object
            if hasattr(correlation_matrix, 'data'):
                corr_df = correlation_matrix.data
            else:
                corr_df = correlation_matrix
            
            # Rename the index
            correlation_matrix_display = corr_df.rename(
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
            # Apply styling after rename
            correlation_matrix_display = correlation_matrix_display.style.map(highlight_correlation, subset=correlation_matrix_display.columns[1:])
            st.dataframe(correlation_matrix_display, column_config=correlation_columns_to_display, hide_index=True, height=600)
        
        with tab_select:
            st.subheader("Feature Selection Tools")
            
            # Monte Carlo Feature Subset Search
            st.write("### Monte Carlo Feature Subset Search")
            X_mc, y_mc = get_features_and_target(data)
            feature_names_mc = X_mc.columns.tolist()

            n_trials = st.number_input("Number of random trials", min_value=10, max_value=100000, value=50, step=10)
            min_features = st.number_input("Minimum features per trial", min_value=3, max_value=len(feature_names_mc)-1, value=8, step=1)
            max_features = st.number_input(
                "Maximum features per trial",
                min_value=min_features+1,
                max_value=len(feature_names_mc),
                value=min(min_features+1, len(feature_names_mc)),
                step=1
            )
            
            if st.button("Run Monte Carlo Search"):
                with st.spinner("Running Monte Carlo feature subset search..."):
                    results_mc = monte_carlo_feature_selection(
                        X_mc, y_mc,
                        model_class=lambda: XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist'),
                        n_trials=int(n_trials),
                        min_features=int(min_features),
                        max_features=int(max_features),
                        random_state=123,
                        cv=10
                    )

                results_mc = sorted(results_mc, key=lambda x: x['mae'])
                best = results_mc[0]
                st.write("Best feature subset:", best['features'])
                st.write(", ".join([f"'{f}'" for f in best['features']]))
                st.write("Best MAE:", best['mae'])

                # Save best features to file
                with open('data_files/f1_position_model_best_features_monte_carlo.txt', 'w') as f:
                    f.write('\n'.join(best['features']))
                    f.write(f"\nBest MAE: {best['mae']:.4f}\n")
                st.success("Best features and MAE saved to f1_position_model_best_features_monte_carlo.txt")

                st.subheader("Top 20 Feature Subsets")
                st.dataframe(pd.DataFrame(results_mc[:20]), hide_index=True, column_config={
                    "features": "Feature Subset",
                    "mae": "Mean Absolute Error (MAE)",
                    "rmse": "Root Mean Squared Error (RMSE)",
                    "r2": "R² Score"
                })

                from collections import Counter
                top_features = [f for r in results_mc[:20] for f in r['features']]
                feature_counts = Counter(top_features)
                feature_counts_df = pd.DataFrame(feature_counts.items(), columns=['Feature', 'Appearances']).sort_values(by='Appearances', ascending=False)
                st.subheader("Feature Appearance in Top 20 Subsets")
                st.dataframe(feature_counts_df, hide_index=True, width=600)

            # RFE
            st.write("### Recursive Feature Elimination (RFE)")
            X_rfe, y_rfe = get_features_and_target(data)
            mask_rfe = y_rfe.notnull() & np.isfinite(y_rfe)
            X_rfe, y_rfe = X_rfe[mask_rfe], y_rfe[mask_rfe]
            for col in X_rfe.select_dtypes(include='object').columns:
                X_rfe[col] = X_rfe[col].astype('category').cat.codes
            
            n_features_rfe = st.number_input("Number of features to select (RFE)", min_value=1, max_value=len(X_rfe.columns), value=10, step=1)
            if st.button("Run RFE"):
                with st.spinner("Running RFE..."):
                    selected_features, ranking = run_rfe_feature_selection(X_rfe, y_rfe, n_features_to_select=int(n_features_rfe))
                st.write("Selected features:", selected_features)
                st.dataframe(pd.DataFrame({'Feature': X_rfe.columns, 'Ranking': ranking}).sort_values('Ranking'), width=600, hide_index=True)

                # Save selected features to file
                with open('data_files/f1_position_model_rfe_selected_features.txt', 'w') as f:
                    f.write('\n'.join(selected_features))
                st.success("RFE selected features saved to f1_position_model_rfe_selected_features.txt")

            # Boruta
            st.write("### Boruta Feature Selection")
            X_boruta, y_boruta = get_features_and_target(data)
            mask_boruta = y_boruta.notnull() & np.isfinite(y_boruta)
            X_boruta, y_boruta = X_boruta[mask_boruta], y_boruta[mask_boruta]
            for col in X_boruta.select_dtypes(include='object').columns:
                X_boruta[col] = X_boruta[col].astype('category').cat.codes
            max_iter_boruta = st.number_input("Boruta max iterations", min_value=10, max_value=200, value=50, step=10)
            if st.button("Run Boruta"):
                with st.spinner("Running Boruta..."):
                    selected_features_b, ranking_b = run_boruta_feature_selection(X_boruta, y_boruta, max_iter=int(max_iter_boruta))
                st.write("Selected features:", selected_features_b)
                st.dataframe(pd.DataFrame({'Feature': X_boruta.columns[:len(ranking_b)], 'Ranking': ranking_b}).sort_values('Ranking'))
                st.write("Best feature subset (quoted, comma-delimited):")
                st.write(", ".join([f"'{f}'" for f in selected_features_b]))

                # Save selected features to file
                with open('data_files/f1_position_model_boruta_selected_features.txt', 'w') as f:
                    f.write('\n'.join(selected_features_b))
                st.success("Boruta selected features saved to f1_position_model_boruta_selected_features.txt")

            # RFE to Minimize MAE
            st.write("### RFE to Minimize MAE")
            X_rfe_mae, y_rfe_mae = get_features_and_target(data)
            mask_rfe_mae = y_rfe_mae.notnull() & np.isfinite(y_rfe_mae)
            X_rfe_mae, y_rfe_mae = X_rfe_mae[mask_rfe_mae], y_rfe_mae[mask_rfe_mae]
            min_features_mae = st.number_input("Min features", min_value=1, max_value=len(X_rfe_mae.columns)-1, value=3, step=1)
            max_features_mae = st.number_input(
                "Max features",
                min_value=min_features_mae+1,
                max_value=len(X_rfe_mae.columns),
                value=min(min_features_mae+5, len(X_rfe_mae.columns)),
                step=1
            )
            if st.button("Run RFE to Minimize MAE"):
                with st.spinner("Running RFE to minimize MAE..."):
                    best_features, best_ranking, best_mae, maes = rfe_minimize_mae(X_rfe_mae, y_rfe_mae, min_features=int(min_features_mae), max_features=int(max_features_mae))
                st.write(f"Best MAE: {best_mae:.3f}")
                st.write("Best feature subset:", best_features)
                st.dataframe(pd.DataFrame({'Feature': best_features}))
                st.line_chart(pd.DataFrame(maes, columns=['n_features', 'MAE']).set_index('n_features'))
                st.write("Best feature subset (quoted, comma-delimited):")
                st.write(", ".join([f"'{f}'" for f in best_features]))

                # Save best features to file
                with open('data_files/f1_position_model_rfe_mae_best_features.txt', 'w') as f:
                    f.write('\n'.join(best_features))
                    f.write(f"\nBest MAE: {best_mae:.4f}\n")
                st.success("RFE MAE best features saved to f1_position_model_rfe_mae_best_features.txt")

            # External script runner (feature selection helper)
            st.write("### External Feature Selection Script")
            
            if st.button("Run feature-selection helper", help="Runs the feature_selection_refinement.py script to perform additional feature selection analyses."):
                with st.spinner('Launching feature selection script...'):
                    import subprocess, sys
                    script_path = os.path.join('scripts', 'feature_selection_refinement.py')
                    log_dir = os.path.join('scripts', 'output')
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, 'feature_selection_stdout.log')
                    try:
                        # Capture full stdout/stderr to a log file for post-mortem
                        with open(log_path, 'w', encoding='utf-8') as logfile:
                            proc = subprocess.Popen([sys.executable, script_path], stdout=logfile, stderr=subprocess.STDOUT, text=True)
                            proc.wait()

                        # Read final summary from feature_selection_report.txt if available
                        report_path = os.path.join(log_dir, 'feature_selection_report.txt')
                        if os.path.exists(report_path):
                            try:
                                rpt = open(report_path, 'r', encoding='utf-8').read()
                                st.subheader('Feature selection summary')
                                st.code(rpt)
                            except Exception:
                                st.write('Feature selection completed; could not read summary report.')
                        else:
                            st.write('Feature selection completed; no summary report found.')

                        if proc.returncode == 0:
                            st.success('Feature selection script completed successfully.')
                        else:
                            st.error(f'Feature selection script exited with code {proc.returncode}. See full log below for details.')
                            # Show tail of the log to help debugging
                            try:
                                with open(log_path, 'r', encoding='utf-8', errors='ignore') as lf:
                                    lines = lf.readlines()[-200:]
                                with st.expander('Show last 200 lines of full run log'):
                                    for l in lines:
                                        st.text(l.rstrip())
                            except Exception:
                                st.write('Could not read log file')
                    except Exception as e:
                        st.error(f'Failed to run feature selection script: {e}')

            # Show outputs if available
            out_dir = os.path.join('scripts', 'output')
            if os.path.exists(out_dir):
                if os.path.exists(os.path.join(out_dir, 'boruta_selected.txt')):
                    st.subheader('Boruta Selected Features')
                    try:
                        with open(os.path.join(out_dir, 'boruta_selected.txt'), 'r', encoding='utf-8') as f:
                            boruta_lines = [l.strip() for l in f.readlines() if l.strip()]
                        st.write(boruta_lines[:100])
                        try:
                            # Render clickable icon + download link (small text file)
                            import base64
                            bpath = os.path.join(out_dir, 'boruta_selected.txt')
                            with open(bpath, 'rb') as bf:
                                bdata = bf.read()
                            file_uri = 'data:text/plain;base64,' + base64.b64encode(bdata).decode('ascii')
                            icon_path_local = os.path.join('data_files', 'csv_icon.png')
                            fallback_icon = os.path.join('data_files', 'favicon.png')
                            chosen_icon = icon_path_local if os.path.exists(icon_path_local) else (fallback_icon if os.path.exists(fallback_icon) else None)
                            img_tag = ''
                            if chosen_icon is not None:
                                try:
                                    with open(chosen_icon, 'rb') as ifh:
                                        img_b64 = base64.b64encode(ifh.read()).decode('ascii')
                                    img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;margin-right:10px;vertical-align:middle;border-radius:6px;">'
                                except Exception:
                                    img_tag = ''
                            html = (
                                f'<div style="display:flex;align-items:center;margin:6px 0;">'
                                f'<a download="boruta_selected.txt" href="{file_uri}" '
                                f'style="display:flex;align-items:center;padding:6px 12px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
                                f'{img_tag}'
                                f'<span style="color:#fff;">Download boruta_selected.txt</span>'
                                f'</a></div>'
                            )
                            st.markdown(html, unsafe_allow_html=True)
                        except Exception:
                            try:
                                with open(os.path.join(out_dir, 'boruta_selected.txt'), 'rb') as fbin:
                                    st.download_button('Download Boruta list', fbin, file_name='boruta_selected.txt')
                            except Exception:
                                st.write('Could not read boruta_selected.txt')
                    except Exception:
                        st.write('Could not read boruta_selected.txt')

                if os.path.exists(os.path.join(out_dir, 'shap_ranking.txt')):
                    st.subheader('SHAP Ranking (top 20)')
                    try:
                        # Prefer CSV-style read, but fall back to parsing plain text
                        shp_path = os.path.join(out_dir, 'shap_ranking.txt')
                        try:
                            # Try to auto-detect separator (handles CSV or tab-delimited)
                            try:
                                df_shap = pd.read_csv(shp_path, sep=None, engine='python')
                            except Exception:
                                df_shap = pd.read_csv(shp_path)
                        except Exception:
                            # Fallback: parse as plain text lines into Feature / SHAP value
                            def _is_number(x):
                                try:
                                    float(x)
                                    return True
                                except Exception:
                                    return False

                            rows = []
                            with open(shp_path, 'r', encoding='utf-8', errors='ignore') as f:
                                for raw in f:
                                    s = raw.strip()
                                    if not s:
                                        continue
                                    # Try common separators first
                                    if ',' in s:
                                        parts = [p.strip() for p in s.split(',') if p.strip()]
                                    elif '\t' in s:
                                        parts = [p.strip() for p in s.split('\t') if p.strip()]
                                    elif ' - ' in s:
                                        parts = [p.strip() for p in s.split(' - ') if p.strip()]
                                    elif ':' in s and s.count(':') == 1:
                                        parts = [p.strip() for p in s.split(':') if p.strip()]
                                    else:
                                        parts = s.split()

                                    if len(parts) >= 2 and _is_number(parts[-1]):
                                        feature = ' '.join(parts[:-1]).strip()
                                        val = parts[-1]
                                    elif len(parts) >= 2:
                                        # Last part may be the value, even if not numeric
                                        feature = ' '.join(parts[:-1]).strip()
                                        val = parts[-1]
                                    else:
                                        # Can't split confidently; put whole line in Feature
                                        feature = s
                                        val = ''

                                    rows.append({'Feature': feature, 'SHAP': val})

                            df_shap = pd.DataFrame(rows)
                            # Attempt to coerce SHAP values to numeric where possible
                            if 'SHAP' in df_shap.columns:
                                df_shap['SHAP'] = pd.to_numeric(df_shap['SHAP'], errors='coerce')

                        # Normalize common column names to nice display names
                        if isinstance(df_shap, pd.DataFrame):
                            cols_lower = [c.lower() for c in df_shap.columns]
                            if 'feature' in cols_lower and 'mean_abs_shap' in cols_lower:
                                # map to consistent names
                                mapping = {df_shap.columns[cols_lower.index('feature')]: 'Feature',
                                           df_shap.columns[cols_lower.index('mean_abs_shap')]: 'SHAP'}
                                df_shap = df_shap.rename(columns=mapping)[['Feature', 'SHAP']]
                            elif len(df_shap.columns) >= 2:
                                # Prefer first two columns
                                df_shap = df_shap.iloc[:, :2]
                                df_shap.columns = ['Feature', 'SHAP']
                        # Display top 20 rows (if available). Guard height calculation.
                        height = get_dataframe_height(df_shap)
                        try:
                            st.dataframe(df_shap.head(20), height=height, hide_index=True, width=600)
                        except Exception:
                            st.dataframe(df_shap.head(20), hide_index=True, width=600, height=height)
                        try:
                            import base64
                            with open(shp_path, 'rb') as sf:
                                sdata = sf.read()
                            file_uri = 'data:text/plain;base64,' + base64.b64encode(sdata).decode('ascii')
                            icon_path_local = os.path.join('data_files', 'csv_icon.png')
                            fallback_icon = os.path.join('data_files', 'favicon.png')
                            chosen_icon = icon_path_local if os.path.exists(icon_path_local) else (fallback_icon if os.path.exists(fallback_icon) else None)
                            img_tag = ''
                            if chosen_icon is not None:
                                try:
                                    with open(chosen_icon, 'rb') as ifh:
                                        img_b64 = base64.b64encode(ifh.read()).decode('ascii')
                                    img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;margin-right:10px;vertical-align:middle;border-radius:6px;">'
                                except Exception:
                                    img_tag = ''
                            html = (
                                f'<div style="display:flex;align-items:center;margin:6px 0;">'
                                f'<a download="shap_ranking.txt" href="{file_uri}" '
                                f'style="display:flex;align-items:center;padding:6px 12px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
                                f'{img_tag}'
                                f'<span style="color:#fff;">Download SHAP ranking</span>'
                                f'</a></div>'
                            )
                            st.markdown(html, unsafe_allow_html=True)
                        except Exception:
                            try:
                                with open(shp_path, 'rb') as fbin:
                                    st.download_button('Download SHAP ranking', fbin, file_name='shap_ranking.txt')
                            except Exception:
                                st.write('Could not read shap_ranking.txt')
                    except Exception as e:
                        st.write('Could not read shap_ranking.txt')
                        try:
                            st.write('Error:', str(e))
                            # Show a small preview of the file to help debugging
                            preview_path = os.path.abspath(shp_path)
                            with open(preview_path, 'r', encoding='utf-8', errors='ignore') as pf:
                                lines = pf.readlines()[:50]
                            with st.expander('Preview of shap_ranking.txt (first 50 lines)'):
                                for l in lines:
                                    st.text(l.rstrip())
                        except Exception as e2:
                            st.write('Also could not read file preview:', str(e2))

                if os.path.exists(os.path.join(out_dir, 'correlated_pairs.csv')):
                    st.subheader('Highly Correlated Pairs (>0.95)')
                    try:
                        df_corr = pd.read_csv(os.path.join(out_dir, 'correlated_pairs.csv'))
                        height = get_dataframe_height(df_corr)
                        st.dataframe(df_corr.head(50), hide_index=True, width=800, height=height)
                        # Render clickable icon + HTML download (data-URI) for correlated_pairs.csv
                        try:
                            import base64
                            csv_path = os.path.join(out_dir, 'correlated_pairs.csv')
                            with open(csv_path, 'rb') as fbin:
                                data_bytes = fbin.read()
                            file_uri = 'data:text/csv;base64,' + base64.b64encode(data_bytes).decode('ascii')
                            icon_path_local = os.path.join('data_files', 'csv_icon.png')
                            fallback_icon = os.path.join('data_files', 'favicon.png')
                            chosen_icon = icon_path_local if os.path.exists(icon_path_local) else (fallback_icon if os.path.exists(fallback_icon) else None)
                            img_tag = ''
                            if chosen_icon is not None:
                                try:
                                    with open(chosen_icon, 'rb') as ifh:
                                        img_b64 = base64.b64encode(ifh.read()).decode('ascii')
                                    img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;margin-right:10px;vertical-align:middle;border-radius:6px;">'
                                except Exception:
                                    img_tag = ''
                            html = (
                                f'<div style="display:flex;align-items:center;margin:6px 0;">'
                                f'<a download="correlated_pairs.csv" href="{file_uri}" '
                                f'style="display:flex;align-items:center;padding:6px 12px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
                                f'{img_tag}'
                                f'<span style="color:#fff;">Download correlated_pairs.csv</span>'
                                f'</a></div>'
                            )
                            st.markdown(html, unsafe_allow_html=True)
                        except Exception:
                            try:
                                with open(os.path.join(out_dir, 'correlated_pairs.csv'), 'rb') as fbin:
                                    st.download_button('Download correlated pairs', fbin, file_name='correlated_pairs.csv')
                            except Exception:
                                st.write('Could not read correlated_pairs.csv')
                    except Exception:
                        st.write('Could not read correlated_pairs.csv')

                # Exporter outputs (CSV summary and HTML report)
                summary_csv = os.path.join(out_dir, 'feature_selection_summary.csv')
                summary_html = os.path.join(out_dir, 'feature_selection_report.html')
                st.write('### Exported Summaries')
                if os.path.exists(summary_csv):
                    try:
                        # Render HTML clickable icon + data-URI for the summary CSV
                        import base64
                        csv_path_local = summary_csv
                        with open(csv_path_local, 'rb') as fbin:
                            data_bytes = fbin.read()
                        file_uri = 'data:text/csv;base64,' + base64.b64encode(data_bytes).decode('ascii')
                        icon_path_local = os.path.join('data_files', 'csv_icon.png')
                        fallback_icon = os.path.join('data_files', 'favicon.png')
                        chosen_icon = icon_path_local if os.path.exists(icon_path_local) else (fallback_icon if os.path.exists(fallback_icon) else None)
                        img_tag = ''
                        if chosen_icon is not None:
                            try:
                                with open(chosen_icon, 'rb') as ifh:
                                    img_b64 = base64.b64encode(ifh.read()).decode('ascii')
                                img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;margin-right:10px;vertical-align:middle;border-radius:6px;">'
                            except Exception:
                                img_tag = ''
                        html = (
                            f'<div style="display:flex;align-items:center;margin:6px 0;">'
                            f'<a download="feature_selection_summary.csv" href="{file_uri}" '
                            f'style="display:flex;align-items:center;padding:6px 12px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
                            f'{img_tag}'
                            f'<span style="color:#fff;">Download summary (CSV)</span>'
                            f'</a></div>'
                        )
                        st.markdown(html, unsafe_allow_html=True)
                    except Exception:
                        try:
                            with open(summary_csv, 'rb') as fbin:
                                st.download_button('Download summary (CSV)', fbin, file_name='feature_selection_summary.csv')
                        except Exception:
                            st.write('Could not read feature_selection_summary.csv')
                if os.path.exists(summary_html):
                    try:
                        st.write(f"HTML report available: {os.path.basename(summary_html)}")
                        try:
                            import base64
                            with open(summary_html, 'rb') as rh:
                                rdata = rh.read()
                            file_uri = 'data:text/html;base64,' + base64.b64encode(rdata).decode('ascii')
                            icon_path_local = os.path.join('data_files', 'csv_icon.png')
                            fallback_icon = os.path.join('data_files', 'favicon.png')
                            chosen_icon = icon_path_local if os.path.exists(icon_path_local) else (fallback_icon if os.path.exists(fallback_icon) else None)
                            img_tag = ''
                            if chosen_icon is not None:
                                try:
                                    with open(chosen_icon, 'rb') as ifh:
                                        img_b64 = base64.b64encode(ifh.read()).decode('ascii')
                                    img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;margin-right:10px;vertical-align:middle;border-radius:6px;">'
                                except Exception:
                                    img_tag = ''
                            html = (
                                f'<div style="display:flex;align-items:center;margin:6px 0;">'
                                f'<a download="{os.path.basename(summary_html)}" href="{file_uri}" '
                                f'style="display:flex;align-items:center;padding:6px 12px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
                                f'{img_tag}'
                                f'<span style="color:#fff;">Download report (HTML)</span>'
                                f'</a></div>'
                            )
                            st.markdown(html, unsafe_allow_html=True)
                        except Exception:
                            with open(summary_html, 'rb') as fbin:
                                st.download_button('Download report (HTML)', fbin, file_name='feature_selection_report.html')
                    except Exception:
                        st.write('Could not read feature_selection_report.html')

                # Regenerate exporters on demand
                if st.button('Regenerate CSV/HTML exporters'):
                    with st.spinner('Generating CSV summary and HTML report...'):
                        script_path = os.path.join('scripts', 'export_feature_selection.py')
                        try:
                            proc = subprocess.run([sys.executable, script_path], check=False, capture_output=True, text=True)
                            if proc.returncode == 0:
                                st.success('Exporters generated successfully.')
                                if proc.stdout:
                                    st.text(proc.stdout)
                            else:
                                st.error(f'Exporter exited with code {proc.returncode}')
                                if proc.stdout:
                                    st.text(proc.stdout)
                                if proc.stderr:
                                    st.text(proc.stderr)
                        except Exception as e:
                            st.error(f'Failed to run exporter: {e}')

                # Prefer a nicely-formatted Markdown report if available
                md_report = os.path.join(out_dir, 'feature_selection_report.md')
                txt_report = os.path.join(out_dir, 'feature_selection_report.txt')
                if os.path.exists(md_report):
                    st.subheader('Feature Selection Report')
                    try:
                        rpt_md = open(md_report, 'r', encoding='utf-8').read()
                        st.markdown(rpt_md)
                        try:
                            import base64
                            with open(md_report, 'rb') as mf:
                                mdata = mf.read()
                            file_uri = 'data:text/markdown;base64,' + base64.b64encode(mdata).decode('ascii')
                            icon_path_local = os.path.join('data_files', 'csv_icon.png')
                            fallback_icon = os.path.join('data_files', 'favicon.png')
                            chosen_icon = icon_path_local if os.path.exists(icon_path_local) else (fallback_icon if os.path.exists(fallback_icon) else None)
                            img_tag = ''
                            if chosen_icon is not None:
                                try:
                                    with open(chosen_icon, 'rb') as ifh:
                                        img_b64 = base64.b64encode(ifh.read()).decode('ascii')
                                    img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;margin-right:10px;vertical-align:middle;border-radius:6px;">'
                                except Exception:
                                    img_tag = ''
                            html = (
                                f'<div style="display:flex;align-items:center;margin:6px 0;">'
                                f'<a download="{os.path.basename(md_report)}" href="{file_uri}" '
                                f'style="display:flex;align-items:center;padding:6px 12px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
                                f'{img_tag}'
                                f'<span style="color:#fff;">Download report (MD)</span>'
                                f'</a></div>'
                            )
                            st.markdown(html, unsafe_allow_html=True)
                        except Exception:
                            with open(md_report, 'rb') as fbin:
                                st.download_button('Download report (MD)', fbin, file_name='feature_selection_report.md')
                    except Exception:
                        st.write('Could not read feature_selection_report.md')
                elif os.path.exists(txt_report):
                    st.subheader('Feature Selection Report')
                    try:
                        rpt = open(txt_report, 'r', encoding='utf-8').read()
                        st.code(rpt)
                        try:
                            import base64
                            with open(txt_report, 'rb') as tf:
                                tdata = tf.read()
                            file_uri = 'data:text/plain;base64,' + base64.b64encode(tdata).decode('ascii')
                            icon_path_local = os.path.join('data_files', 'csv_icon.png')
                            fallback_icon = os.path.join('data_files', 'favicon.png')
                            chosen_icon = icon_path_local if os.path.exists(icon_path_local) else (fallback_icon if os.path.exists(fallback_icon) else None)
                            img_tag = ''
                            if chosen_icon is not None:
                                try:
                                    with open(chosen_icon, 'rb') as ifh:
                                        img_b64 = base64.b64encode(ifh.read()).decode('ascii')
                                    img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;margin-right:10px;vertical-align:middle;border-radius:6px;">'
                                except Exception:
                                    img_tag = ''
                            html = (
                                f'<div style="display:flex;align-items:center;margin:6px 0;">'
                                f'<a download="{os.path.basename(txt_report)}" href="{file_uri}" '
                                f'style="display:flex;align-items:center;padding:6px 12px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
                                f'{img_tag}'
                                f'<span style="color:#fff;">Download report</span>'
                                f'</a></div>'
                            )
                            st.markdown(html, unsafe_allow_html=True)
                        except Exception:
                            with open(txt_report, 'rb') as fbin:
                                st.download_button('Download report', fbin, file_name='feature_selection_report.txt')
                    except Exception:
                        st.write('Could not read feature_selection_report.txt')
        
        with tab_position:
            from pathlib import Path
            import re, datetime

            OUT_DIR = Path('scripts') / 'output'
            report_path = OUT_DIR / 'position_group_analysis_report.html'

            if report_path.exists():
                # Try to extract the generated timestamp from the HTML; fallback to file mtime
                try:
                    html = report_path.read_text(encoding='utf-8')
                    m = re.search(r'Generated:\s*([^<]+)</p>', html)
                    if m:
                        ts = m.group(1).strip()
                    else:
                        ts = datetime.datetime.fromtimestamp(report_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    ts = datetime.datetime.fromtimestamp(report_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')

                st.header('Position Group Analysis')
                st.write(f'Generated: {ts}')

                # MAE by season: only show when multiple seasons are present
                mae_csv = OUT_DIR / 'mae_by_season.csv'
                if mae_csv.exists():
                    try:
                        mae_df = pd.read_csv(mae_csv)
                        if 'season' in mae_df.columns and mae_df['season'].nunique() > 1:
                            mae_img = OUT_DIR / 'mae_trends.png'
                            if mae_img.exists():
                                st.subheader('MAE by Season')
                                st.image(str(mae_img), use_container_width=True)
                    except Exception:
                        pass

                # Notes (render in a shaded info box)
                st.info(
                    """
                    **Color scale**: darker/warmer colors indicate larger average absolute error.

                    **Missing cells**: blank or neutral color means insufficient data (no races for that pair).

                    **Sample size**: confidence intervals are empirical percentiles computed only when a group has at least 5 residuals.
                    
                    **Interpretation**: cells with darker colors indicate that the model has higher prediction errors for that driver/constructor at that circuit, suggesting potential areas for model improvement or unique performance characteristics.
                    """
                )

                # Heatmaps
                for img_name, title in [('heatmap_driver_by_circuit.png', 'Driver x Circuit heatmap'),
                                        ('heatmap_constructor_by_circuit.png', 'Constructor x Circuit heatmap')]:
                    img_path = OUT_DIR / img_name
                    if img_path.exists():
                        st.subheader(title)
                        st.image(str(img_path), width=1000)

                # CSV download buttons (show small icon from `data_files/` if available)
                csv_files = ['mae_by_season.csv', 'confid_int_by_driver_track.csv', 'confid_int_by_driver.csv', 'confid_int_by_constructor.csv']
                icons_dir = Path('data_files')
                csv_icon = icons_dir / 'csv_icon.png'
                pdf_icon = icons_dir / 'pdf_icon.png'
                fallback_icon = icons_dir / 'favicon.png'

                # Render HTML-based download buttons with embedded icons (base64 data-URIs).
                # Fallback to the existing Streamlit download button if something goes wrong.
                import base64
                for fname in csv_files:
                    p = OUT_DIR / fname
                    if not p.exists():
                        continue
                    try:
                        with open(p, 'rb') as fh:
                            data_bytes = fh.read()

                        # Prepare data URI for download link
                        b64_file = base64.b64encode(data_bytes).decode('ascii')
                        file_data_uri = f"data:text/csv;base64,{b64_file}"

                        # Choose icon (csv_icon preferred, fallback to favicon)
                        chosen_icon_path = None
                        if p.suffix.lower() == '.csv' and csv_icon.exists():
                            chosen_icon_path = csv_icon
                        elif p.suffix.lower() == '.pdf' and pdf_icon.exists():
                            chosen_icon_path = pdf_icon
                        elif fallback_icon.exists():
                            chosen_icon_path = fallback_icon

                        img_tag = ''
                        if chosen_icon_path is not None:
                            try:
                                with open(chosen_icon_path, 'rb') as ifh:
                                    img_b64 = base64.b64encode(ifh.read()).decode('ascii')
                                # Larger icon and rounded corners for nicer appearance
                                img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;margin-right:10px;vertical-align:middle;border-radius:6px;">'
                            except Exception:
                                img_tag = ''

                        # HTML for a compact icon + button-like link; wrap the image inside the anchor so it's clickable
                        html = (
                            f'<div style="display:flex;align-items:center;margin:6px 0;">'
                            f'<a download="{fname}" href="{file_data_uri}" '
                            f'style="display:flex;align-items:center;padding:6px 12px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
                            f'{img_tag}'
                            f'<span style="color:#fff;">Download {fname}</span>'
                            f'</a></div>'
                        )

                        st.markdown(html, unsafe_allow_html=True)
                    except Exception:
                        # fallback to Streamlit native button
                        try:
                            with open(p, 'rb') as fh:
                                data_bytes = fh.read()
                            st.download_button(f"Download {fname}", data_bytes, file_name=fname)
                        except Exception:
                            st.write(f'Could not prepare download for {fname}')
            else:
                st.info("Position analysis report not found. Run `python scripts/position_group_analysis.py` to generate outputs.")
        
        with tab_hyper:
            st.subheader("Hyperparameter Tuning")
            
            # Early Stopping Details
            st.write("### Early Stopping Details")
            # Handle different evals_result formats for different XGBoost APIs
            if 'eval' in evals_result and ('absolute_error' in evals_result['eval'] or 'mae' in evals_result['eval']):
                mae_per_round = evals_result['eval']['absolute_error'] if 'absolute_error' in evals_result['eval'] else evals_result['eval']['mae']
            elif 'validation_0' in evals_result and 'mae' in evals_result['validation_0']:
                mae_per_round = evals_result['validation_0']['mae']
            else:
                # Fallback
                mae_per_round = [getattr(model, 'best_score', 0)]
            
            if len(mae_per_round) > 0:
                best_round = int(np.argmin(mae_per_round))
                lowest_mae = mae_per_round[best_round]
                st.write(f"Early stopping occurred at round {best_round + 1} (lowest MAE: {lowest_mae:.4f})")
                st.line_chart(mae_per_round)
            else:
                st.write("Early stopping details not available")

            feature_names_early = preprocessor.get_feature_names_out()
            feature_names_early = [name.replace('num__', '').replace('cat__', '') for name in feature_names_early]

            # Get importances based on model type (early stopping section)
            if hasattr(model, 'get_booster'):  # XGBoost (XGBRegressor)
                importances_dict_early = model.get_booster().get_score(importance_type='weight')
                importances_early = []
                for i, name in enumerate(feature_names_early):
                    importances_early.append(importances_dict_early.get(f'f{i}', 0))
            elif hasattr(model, 'feature_importances_'):  # LightGBM, CatBoost, sklearn models
                importances_early = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):  # CatBoost
                importances_early = model.get_feature_importance()
            else:  # Ensemble or other
                importances_early = [0] * len(feature_names_early)  # Default to zero

            feature_importances_df_early = pd.DataFrame({
                'Feature': feature_names_early,
                'Importance': importances_early,
                'Percentage': np.array(importances_early) / (np.sum(importances_early) or 1) * 100
            }).sort_values(by='Importance', ascending=False)

            top_feature = feature_importances_df_early.iloc[0]
            st.write(f"Most important feature after training: **{top_feature['Feature']}** (Importance: {top_feature['Importance']})")
            st.dataframe(feature_importances_df_early.head(50), hide_index=True, width=800)

            # Hyperparameter Tuning
            st.write("### Run Hyperparameter Tuning")
            
            tuning_method = st.selectbox("Tuning Method", ["Grid Search", "Bayesian Optimization"], key="tuning_method")
            
            if st.button("Start Hyperparameter Tuning"):
                with st.spinner("Running hyperparameter tuning (this may take several minutes)..."):
                    X_hyper, y_hyper = get_features_and_target(data)
                    
                    # Get season for stratified CV
                    season_groups = data.loc[y_hyper.index, 'year'] if 'year' in data.columns else None
                    
                    mask_hyper = y_hyper.notnull() & np.isfinite(y_hyper)
                    X_clean, y_clean = X_hyper[mask_hyper], y_hyper[mask_hyper]
                    season_clean = season_groups[mask_hyper] if season_groups is not None else None
                    
                    if tuning_method == "Grid Search":
                          param_grid = {
                              'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
                              'regressor__max_depth': [3, 4, 5, 6, 7],
                              'regressor__min_child_weight': [1, 3, 5, 7],
                          }
                          pipeline = Pipeline([
                              ('preprocessor', get_preprocessor_position(X_clean)),
                              ('regressor', XGBRegressor(n_estimators=100, random_state=42))
                          ])
                          
                          # Use GroupKFold if season data available, else StratifiedKFold approximation
                          if season_clean is not None:
                              cv = GroupKFold(n_splits=5)
                              groups = season_clean
                          else:
                              cv = 5
                              groups = None
                              
                          grid_search = GridSearchCV(pipeline, param_grid, cv=cv, groups=groups, scoring='neg_mean_absolute_error')
                          grid_search.fit(X_clean, y_clean)
                          st.write("Best params:", grid_search.best_params_)
                          st.write(f"Best MAE: {-grid_search.best_score_:.4f}")
                        
                    elif tuning_method == "Bayesian Optimization":
                        import optuna
                        
                        def objective(trial):
                            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                            max_depth = trial.suggest_int('max_depth', 3, 10)
                            min_child_weight = trial.suggest_int('min_child_weight', 1, 10)

                            pipeline = Pipeline([
                                ('preprocessor', get_preprocessor_position(X_clean)),
                                ('regressor', XGBRegressor(
                                    n_estimators=100,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    min_child_weight=min_child_weight,
                                    random_state=42
                                ))
                            ])
                            
                            # Use GroupKFold for season stratification
                            if season_clean is not None:
                                cv = GroupKFold(n_splits=5)
                                groups = season_clean
                            else:
                                cv = 5
                                groups = None
                                
                            scores = cross_val_score(pipeline, X_clean, y_clean, cv=cv, groups=groups, scoring='neg_mean_absolute_error')
                            return -scores.mean()
                        
                        study = optuna.create_study(direction='minimize')
                        study.optimize(objective, n_trials=50)
                        
                        st.write("Best params:", study.best_params)
                        st.write(f"Best MAE: {study.best_value:.4f}")
                        
                        # Plot optimization history
                        fig = optuna.visualization.plot_optimization_history(study)
                        st.plotly_chart(fig)
        
        with tab_hist:
            st.subheader("Historical Validation")
            
            # Model Evaluation Metrics
            st.write("### Model Evaluation Metrics (Cross-Validation)")
            X_eval, y_eval = get_features_and_target(data)
            mask_eval = y_eval.notnull() & np.isfinite(y_eval)
            X_eval, y_eval = X_eval[mask_eval], y_eval[mask_eval]
            
            # Preprocess the features (handle string columns)
            preprocessor = get_preprocessor_position(X_eval)
            X_eval_prep = preprocessor.fit_transform(X_eval)
            
            # Create a fresh estimator for cross-validation
            model_cv = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=42)
            scores = cross_val_score(model_cv, X_eval_prep, y_eval, cv=5, scoring='neg_mean_squared_error')
            avg_mse = -scores.mean()
            std_mse = scores.std()
            st.write(f"Final Position Model - Cross-validated MSE: {avg_mse:.3f} (± {std_mse:.3f})")

            X_dnf_eval, y_dnf_eval = get_features_and_target_dnf(data)
            mask_dnf = y_dnf_eval.notnull() & np.isfinite(y_dnf_eval)
            X_dnf_eval, y_dnf_eval = X_dnf_eval[mask_dnf], y_dnf_eval[mask_dnf]
            if X_dnf_eval.shape[0] == 0:
                st.warning("No data for DNF evaluation; skipping DNF test/CV metrics.")
            else:
                X_train_dnf, X_test_dnf, y_train_dnf, y_test_dnf = train_test_split(X_dnf_eval, y_dnf_eval, test_size=0.2, random_state=42)
                if X_test_dnf.shape[0] == 0:
                    st.warning("DNF test split is empty; skipping MAE calculation.")
                else:
                    y_pred_dnf_proba = get_dnf_model(CACHE_VERSION).predict_proba(X_test_dnf)[:, 1]
                    mae_dnf = mean_absolute_error(y_test_dnf, y_pred_dnf_proba)
                    st.write(f"Mean Absolute Error (MAE) for DNF Probability (test set): {mae_dnf:.3f}")
                scores_dnf = cross_val_score(get_dnf_model(CACHE_VERSION), X_dnf_eval, y_dnf_eval, cv=5, scoring='roc_auc')
                st.write(f"DNF Model - Cross-validated ROC AUC: {scores_dnf.mean():.3f} (± {scores_dnf.std():.3f})")

            X_sc_eval, y_sc_eval = get_features_and_target_safety_car(safety_cars)
            mask_sc = y_sc_eval.notnull() & np.isfinite(y_sc_eval)
            X_sc_eval, y_sc_eval = X_sc_eval[mask_sc], y_sc_eval[mask_sc]
            if X_sc_eval.shape[0] == 0:
                st.warning("No safety-car data for CV evaluation; skipping safety car CV metrics.")
            else:
                scores_sc = cross_val_score(get_safetycar_model(CACHE_VERSION), X_sc_eval, y_sc_eval, cv=5, scoring='roc_auc')
                st.write(f"Safety Car Model - Cross-validated ROC AUC (unique rows): {scores_sc.mean():.3f} (± {scores_sc.std():.3f})")

            # Model Accuracy for All Races
            st.write("### Model Accuracy Across All Races")
            X_all, y_all = get_features_and_target(data)
            X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

            preprocessor_all = get_preprocessor_position(X_all)
            preprocessor_all.fit(X_train_all)
            X_test_prep_all = preprocessor_all.transform(X_test_all)
            
            # Predict based on model type
            if isinstance(model, xgb.Booster):  # XGBoost
                y_pred_all = model.predict(xgb.DMatrix(X_test_prep_all))
            else:  # LightGBM, CatBoost, sklearn models
                y_pred_all = model.predict(X_test_prep_all)

            results_df_all = X_test_all.copy()
            results_df_all['ActualFinalPosition'] = y_test_all.values
            results_df_all['PredictedFinalPosition'] = y_pred_all
            results_df_all['Error'] = results_df_all['ActualFinalPosition'] - results_df_all['PredictedFinalPosition']

            st.write(f"Mean Squared Error: {mse:.3f}")
            st.write(f"R^2 Score: {r2:.3f}")
            st.write(f"Mean Absolute Error: {mae:.2f}")
            st.write(f"Mean Error: {mean_err:.2f}")

            results_df_pos = pd.DataFrame({
                'Actual': y_test_all.values,
                'Predicted': y_pred_all
            })

            podium_actual_hist = results_df_pos[results_df_pos['Actual'] <= 3]
            points_actual_hist = results_df_pos[results_df_pos['Actual'] <= 10]
            winners_actual_hist = results_df_pos[results_df_pos['Actual'] == 1]

            if len(podium_actual_hist) > 0:
                podium_mae_hist = mean_absolute_error(podium_actual_hist['Actual'], podium_actual_hist['Predicted'])
                st.write(f"MAE for Podium Finishers (1-3): {podium_mae_hist:.3f}")
                
            if len(winners_actual_hist) > 0:
                winner_mae_hist = mean_absolute_error(winners_actual_hist['Actual'], winners_actual_hist['Predicted'])
                st.write(f"MAE for Race Winners: {winner_mae_hist:.3f}")

            if len(points_actual_hist) > 0:
                points_mae_hist = mean_absolute_error(points_actual_hist['Actual'], points_actual_hist['Predicted'])
                st.write(f"MAE for Points Positions (1-10): {points_mae_hist:.3f}")

            st.dataframe(
                results_df_all[['constructorName', 'resultsDriverName', 'ActualFinalPosition', 'PredictedFinalPosition', 'Error']].sort_values(by=['ActualFinalPosition']),
                hide_index=True,
                width=1000,
                column_config={
                    'constructorName': st.column_config.TextColumn("Constructor"),
                    'resultsDriverName': st.column_config.TextColumn("Driver"),
                    'ActualFinalPosition': st.column_config.NumberColumn("Actual", format="%d"),
                    'PredictedFinalPosition': st.column_config.NumberColumn("Predicted", format="%.2f"),
                    'Error': st.column_config.NumberColumn("Error", format="%.2f"),
                }
            )

            st.subheader("Actual vs Predicted Final Position (All Races)")
            st.scatter_chart(results_df_all, x='ActualFinalPosition', y='PredictedFinalPosition', width='stretch')
        
        with tab_debug:
            st.subheader("Debug & Experiments")
                        
            # Bin Count Comparison
            st.write("### Compare Different Bin Counts (q)")
            from feature_lists import high_cardinality_features
            q_values = st.multiselect("Select q values (number of bins)", [2, 3, 4, 5, 6, 7, 8, 9, 10], default=[2, 3, 4, 5, 6, 7, 8, 9, 10])
            
            if st.button("Run Bin Count Comparison"):
                results_bin = []
                for q in q_values:
                    df_bin = data.copy()
                    for col in high_cardinality_features:
                        try:
                            df_bin[f"{col}_bin"] = pd.qcut(df_bin[col], q=q, labels=False, duplicates='drop')
                        except Exception as e:
                            continue
                            
                        X_bin, y_bin = get_features_and_target(df_bin)
                        mask_bin = y_bin.notnull() & np.isfinite(y_bin)
                        X_bin, y_bin = X_bin[mask_bin], y_bin[mask_bin]

                        # Use proper preprocessor instead of naive cat.codes
                        preprocessor_bin = get_preprocessor_position(X_bin)
                        X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)
                        X_train_bin_prep = preprocessor_bin.fit_transform(X_train_bin)
                        X_test_bin_prep = preprocessor_bin.transform(X_test_bin)
                        
                        model_bin = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=42)
                        model_bin.fit(X_train_bin_prep, y_train_bin)
                        y_pred_bin = model_bin.predict(X_test_bin_prep)
                        mae_bin = mean_absolute_error(y_test_bin, y_pred_bin)
                        results_bin.append({'q': q, 'MAE': mae_bin})
                        
                    results_df_bin = pd.DataFrame(results_bin).sort_values('q')
                    st.write("MAE for each bin count (q):")
                    st.dataframe(results_df_bin, hide_index=True)
                    st.line_chart(results_df_bin.set_index('q'))

def leakage_audit_ui():
    """Admin UI to run the temporal leakage audit from Streamlit."""
    try:
        with st.expander("🔍 Run Temporal Leakage Audit (Admin)", expanded=False):
            st.write("Run a heuristics-based audit that checks for features likely to leak future information into training.")
            # Short explainer for users/admins describing what the audit does and its outputs
            with st.expander("About this Leakage Audit", expanded=False):
                st.write(
                    "This audit scans the analysis dataset for features that may leak future or post-event information into training."
                )
                st.write("It applies several heuristics:")
                st.write("- Name-pattern checks (e.g. columns containing 'post', 'after', 'final', 'result', 'total').")
                st.write("- Very high Pearson correlation with targets (abs >= 0.95).")
                st.write("- Per-driver lagged-correlation checks: flags features whose correlation with the *next* race result is substantially higher than with the current result, suggesting future information.")
                st.write("- Safety-car related candidate checks (features mentioning 'safety' or similar).")
                st.write("")
                st.write("Output: a CSV at `leakage_audit_report.csv` with columns: feature, issue, target, value, value2, metric_name, explanation, delta, note.")
                st.write("Recommendation: review flagged features and remove or re-engineer any that use post-race or future information before training models.")
            nrows = st.number_input("Rows to read (0 = all)", min_value=0, value=0)
            run = st.button("Run Leakage Audit")
            if run:
                nr = None if int(nrows) == 0 else int(nrows)
                with st.spinner("Running leakage audit..."):
                    try:
                        report_df = audit_temporal_leakage.run_audit(nrows=nr)
                        if report_df is None or report_df.empty:
                            st.success("No suspicious features found by heuristics.")
                        else:
                            st.success(f"Audit finished: {len(report_df)} items")
                            # Rename columns for a friendly UI display
                            ui_map = {
                                'feature': 'Feature',
                                'issue_type': 'Issue',
                                'target': 'Target',
                                'metric': 'Value',
                                'metric2': 'Value2',
                                'metric_name': 'Metric Name',
                                'explanation': 'Explanation',
                                'diff': 'Delta',
                                'extra_info': 'Note'
                            }
                            display_df = report_df.rename(columns=ui_map)

                            st.dataframe(display_df, hide_index=True, width='stretch', column_config={
                                'Feature': st.column_config.TextColumn("Feature"),
                                'Issue': st.column_config.TextColumn("Issue"),
                                'Target': st.column_config.TextColumn("Target"),
                                'Value': st.column_config.NumberColumn("Value", format="%.6f"),
                                'Value2': st.column_config.NumberColumn("Value2", format="%.6f"),
                                'Metric Name': st.column_config.TextColumn("Metric Name"),
                                'Explanation': st.column_config.TextColumn("Explanation"),
                                'Delta': st.column_config.NumberColumn("Delta", format="%.6f"),
                                'Note': st.column_config.TextColumn("Note")
                            })

                            # Prepare downloadable CSV with the friendlier headers
                            csv_df = report_df.rename(columns={
                                'feature': 'feature',
                                'issue_type': 'issue',
                                'target': 'target',
                                'metric': 'value',
                                'metric2': 'value2',
                                'metric_name': 'metric_name',
                                'explanation': 'explanation',
                                'diff': 'delta',
                                'extra_info': 'note'
                            })
                            csv = csv_df.to_csv(index=False)
                            try:
                                # Use HTML clickable icon + download for leakage audit CSV
                                import base64
                                tdata = csv.encode('utf-8') if isinstance(csv, str) else csv
                                file_uri = 'data:text/csv;base64,' + base64.b64encode(tdata).decode('ascii')
                                icon_path_local = os.path.join('data_files', 'csv_icon.png')
                                fallback_icon = os.path.join('data_files', 'favicon.png')
                                chosen_icon = icon_path_local if os.path.exists(icon_path_local) else (fallback_icon if os.path.exists(fallback_icon) else None)
                                img_tag = ''
                                if chosen_icon is not None:
                                    try:
                                        with open(chosen_icon, 'rb') as ifh:
                                            img_b64 = base64.b64encode(ifh.read()).decode('ascii')
                                        img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;margin-right:10px;vertical-align:middle;border-radius:6px;">'
                                    except Exception:
                                        img_tag = ''
                                html = (
                                    f'<div style="display:flex;align-items:center;margin:6px 0;">'
                                    f'<a download="leakage_audit_report.csv" href="{file_uri}" '
                                    f'style="display:flex;align-items:center;padding:6px 12px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
                                    f'{img_tag}'
                                    f'<span style="color:#fff;">Download CSV</span>'
                                    f'</a></div>'
                                )
                                st.markdown(html, unsafe_allow_html=True)
                            except Exception:
                                # Fallback to plain download button
                                st.download_button("Download CSV", csv, file_name='leakage_audit_report.csv')
                    except Exception as e:
                        st.error(f"Audit failed: {e}")
    except Exception:
        # If Streamlit not available or UI errors, fail silently
        pass

with tab6:
    # Force immediate render to test if tab executes
    st.write("Tab 6 START")
    st.header("Data & Debug Tools")
    # Split Data & Debug into subtabs: Raw Data, Temporal Leakage Audit, Hyperparameter Tuning
    raw_tab, audit_tab, tuning_tab = st.tabs(["Raw Data", "Temporal Leakage Audit", "Hyperparameter Tuning"])#, "Position Analysis"])

    with raw_tab:
        st.write("View the complete unfiltered dataset.")
        if st.checkbox('Show Raw Data', value=False):
            st.write(f"Total number of results: {len(data):,d}")
            st.dataframe(data, column_config=columns_to_display,
                hide_index=True, width='stretch', height=600)

    with audit_tab:
        st.write("Run heuristics-based checks for features that may leak future information into models.")
        # with st.expander("Run Temporal Leakage Audit (Admin)", expanded=False):
        try:
            leakage_audit_ui()
        except Exception as e:
            st.error(f"Leakage audit UI failed to render: {e}")

    with tuning_tab:
        st.write("Run basic hyperparameter tuning (GridSearch) on the full dataset.")
        if st.checkbox("Run Hyperparameter Tuning (subtab)"):
            X, y = get_features_and_target(data)
            param_grid = {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [3, 4, 5],
                'regressor__learning_rate': [0.05, 0.1, 0.2],
                'regressor__reg_alpha': [0, 0.1, 0.3],           # L1 regularization
                'regressor__colsample_bytree': [0.6, 0.8, 1.0],  # Sample % of features per tree
                'regressor__colsample_bylevel': [0.6, 0.8, 1.0], # Sample % of features per tree level
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

add_betting_oracle_footer()