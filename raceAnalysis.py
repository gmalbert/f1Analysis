import datetime
import pandas as pd
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
#import re

#variable_to_change = "helloWorld123"
#variable_changed = re.sub( r"([A-Z])|([0-9]+)", r" \1\2", variable_to_change).strip()

#print(f"Updated Variable name {variable_changed.title()}")

##### to do: modify variable names for display

def reset_filters():
    # Assuming you have filters stored in session state
    for key in st.session_state.keys():
        if key.startswith('filter_'):
            st.session_state[key] = None  # or any default value

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
    'averagePracticePosition': 'Average Practice Pos.', 
    'lastFPPositionNumber': 'Last FP Pos.', 
    'resultsQualificationPositionNumber': 'Qual. Pos.', 
    'q1End': 'Out at Q1', 
    'q2End': 'Out at Q2', 
    'q3Top10': 'Q3 Top 10'}

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year
raceNoEarlierThan = current_year - 10

exclusionList = ['grandPrixRaceId', 'raceId_results']

@st.cache_data
def load_data_schedule(nrows):
    raceSchedule = pd.read_json(path.join(DATA_DIR, 'f1db-races.json'))
    grandPrix = pd.read_json(path.join(DATA_DIR, 'f1db-grands-prix.json'))
    raceSchedule = raceSchedule.merge(grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_grandPrix', '_schedule'])
    return raceSchedule

raceSchedule = load_data_schedule(10000)

schedule_columns_to_display = {
    'round': st.column_config.NumberColumn("Round", format="%d"),
    'fullName': st.column_config.TextColumn("Name"),
    'date': st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
    'time': st.column_config.TimeColumn("Time", format="HH:MM"),
    'courseLength': st.column_config.NumberColumn("Course Length (km)", format="%d"),
    'laps': st.column_config.NumberColumn("Number of Laps", format="%d"),
    'turns': st.column_config.NumberColumn("Number of Turns", format="%d"),
    'distance': st.column_config.NumberColumn("Distance (km)", format="%d"),
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
    weather = pd.read_csv(path.join(DATA_DIR, 'f1WeatherData_Grouped.csv'), sep='\t', nrows=nrows, usecols=['grandPrixId', 'short_date', 'average_temp', 'total_precipitation', 'average_humidity', 'average_wind_speed'])
    grandPrix = pd.read_json(path.join(DATA_DIR, 'f1db-grands-prix.json'))
    weather_with_grandprix = pd.merge(weather, grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_weather', '_grandPrix'])
    return weather_with_grandprix

weatherData = load_weather_data(10000)

weather_columns_to_display = {
    'short_date': st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
    'average_temp': st.column_config.NumberColumn("Average Temperature (F)", format="%d"),
    'total_precipitation': st.column_config.NumberColumn("Precipitation (in)", format="%d"),
    'average_humidity': st.column_config.NumberColumn("Average Humidity (%)", format="%d"),
    'average_wind_speed': st.column_config.NumberColumn("Average Wind Speed (mph)", format="%d"),
    'grandPrixId': None,
    'countryId': None,
    'abbreviation': None,
    'shortName': None,
    'name' : None,
    'fullName': None,
    'id': None, 
    'totalRacesHeld' : None
}

st.image(path.join(DATA_DIR, 'formula1_logo.png'))
st.title(f'F1 Races from {raceNoEarlierThan} to {current_year}')

columns_to_display = {'grandPrixYear': st.column_config.NumberColumn("Year", format="%d"),
    'grandPrixName': st.column_config.TextColumn("Grand Prix"),
    'constructorName': st.column_config.TextColumn("Constructor"),
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
    'short_date': None,
    'raceId_results': None,
    'grandPrixRaceId': None,
    'DNF': st.column_config.CheckboxColumn("DNF"),
    'averagePracticePosition': st.column_config.NumberColumn(
        "Avg Practice Pos.", format="%d", min_value=1, max_value=20, step=1, default=1),
    'lastFPPositionNumber': st.column_config.NumberColumn(
        "Last FP Pos.", format="%d", min_value=1, max_value=20, step=1, default=1),
    'resultsQualificationPositionNumber': st.column_config.NumberColumn(
        "Qual. Pos.", format="%d", min_value=1, max_value=20, step=1, default=1),
    'q1End': st.column_config.CheckboxColumn("Out at Q1"),
    'q2End': st.column_config.CheckboxColumn("Out at Q2"),
    'q3Top10': st.column_config.CheckboxColumn("Q3 Top 10")
}

next_race_columns_to_display = {
    'date': st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
    'time': st.column_config.TextColumn("Time"),
    'fullName': st.column_config.TextColumn("Grand Prix"),
    'courseLength': st.column_config.TextColumn("Course Length (km)"),
    'turns': st.column_config.TextColumn("Number of Turns"),
    'laps': st.column_config.TextColumn("Number of Laps")    

}

@st.cache_data
def load_data(nrows):
    fullResults = pd.read_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), sep='\t', nrows=nrows, usecols=['grandPrixYear', 'grandPrixName', 'resultsDriverName', 'resultsPodium', 'resultsTop5', 'resultsTop10', 'constructorName',  'resultsStartingGridPositionNumber', 'resultsFinalPositionNumber', 
    'positionsGained', 'short_date', 'raceId_results', 'grandPrixRaceId', 'DNF', 'averagePracticePosition', 'lastFPPositionNumber', 'resultsQualificationPositionNumber', 'q1End', 'q2End', 'q3Top10'])
    
    return fullResults

data = load_data(10000)

data['averagePracticePosition'] = data['averagePracticePosition'].round(2)

column_names = data.columns.tolist()
column_names.sort()

# Convert columns to appropriate types to allow for NaN values
data['resultsStartingGridPositionNumber'] = data['resultsStartingGridPositionNumber'].astype('Int64')
data['resultsFinalPositionNumber'] = data['resultsFinalPositionNumber'].astype('Int64')
data['positionsGained'] = data['positionsGained'].astype('Int64')
data['averagePracticePosition'] = data['averagePracticePosition'].astype('Float64')
data['lastFPPositionNumber'] = data['lastFPPositionNumber'].astype('Int64')
data['resultsQualificationPositionNumber'] = data['resultsQualificationPositionNumber'].astype('Int64')
data['short_date'] = pd.to_datetime(data['short_date'])

if st.checkbox('Filter Results'):
    # Create a dictionary to store selected filters for multiple columns
    filters = {}

    # Iterate over the columns to display and create a filter for each
    for column in column_names:

        for old_column, new_column in column_rename_for_filter.items():
            if column == old_column: 
                column_friendly_name = new_column
        
        if is_numeric_dtype(data[column]) and (data[column].dtype in ('np.int64', 'np.float64', 'Int64', 'int64', 'Float64') ):

            # Do not display if the column is in the exclusion list noted at the top of the file
            if column not in exclusionList:
                min_val, max_val = int(data[column].min()), int(data[column].max())           
                                
                selected_range = st.sidebar.slider(
                    f"Filter by {column_friendly_name}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=1,
                    key=f"range_filter_{column}"
            )
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
                f"Filter by {column_friendly_name}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                format="YYYY-MM-DD",
                key=f"range_filter_{column}"
            )
            filters[column] = selected_range

        elif data[column].dtype ==bool:
            
           
            selected_value = st.sidebar.checkbox(
                f"{column_friendly_name}",
                value=False,
                key=f"checkbox_filter_{column}"
            )
            if selected_value:
                filters[column] = True
        else:
            unique_values = data[column].dropna().unique().tolist()
            unique_values.sort()
            unique_values.insert(0, ' All')  # Add an option to select all values

            for old_column, new_column in column_rename_for_filter.items():
                    if column == old_column:
                        column_friendly_name = new_column

            if column not in (exclusionList):
                selected_value = st.sidebar.selectbox(
                    f"Filter by {column_friendly_name}",
                    unique_values,
                    key=f"filter_{column}"
            )

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

    st.write(f"Number of filtered results: {len(filtered_data)}")
    st.dataframe(filtered_data, column_config=columns_to_display, column_order=['grandPrixYear', 'grandPrixName', 'constructorName', 'resultsDriverName', 'resultsPodium', 'resultsTop5',
         'resultsTop10','resultsStartingGridPositionNumber','resultsFinalPositionNumber','positionsGained', 'DNF', 'resultsQualificationPositionNumber', 'q1End', 'q2End', 'q3Top10', 'averagePracticePosition',  'lastFPPositionNumber'], hide_index=True, width=2400, height=600)

    ##st.button("Clear multiselect", on_click=clear_multi)

#if st.button("Reset Filters"):
#    reset_filters()
#    st.experimental_rerun()

    # Add visualizations for the filtered data
    st.subheader("Positions Gained")
    st.line_chart(filtered_data, x='short_date', x_label='Date', y='positionsGained', y_label='Positions Gained', use_container_width=True)
    st.scatter_chart(filtered_data, x='short_date', x_label='Date', y='positionsGained', y_label='Positions Gained', use_container_width=True)

if st.checkbox(f"Show {current_year} Schedule"):
    st.title(f"{current_year} Races:")
    
    raceSchedule = raceSchedule[raceSchedule['year'] == current_year]
    st.write(f"Total number of races: {len(raceSchedule)}")
    st.dataframe(raceSchedule, column_config=schedule_columns_to_display,
        hide_index=True,  width=800, height=600, column_order=['round', 'fullName', 'date', 'time', 
        'circuitType', 'courseLength', 'laps', 'turns', 'distance', 'totalRacesHeld'])


if st.checkbox("Show Next Race"):

    st.subheader("Next Race:")
    nextRace = raceSchedule[raceSchedule['date'] >= datetime.datetime.now()]
    nextRace['date']= nextRace['date'].dt.strftime('%Y-%m-%d')
    nextRace = nextRace.sort_values(by=['date', 'time'], ascending=[True, True]).head(1)
    st.dataframe(nextRace, column_config=next_race_columns_to_display, hide_index=True, 
        column_order=['date', 'time', 'fullName', 'courseLength', 'turns', 'laps'])


    # Limit detailsOfNextRace by the grandPrixId of the next race
    next_race_id = nextRace['grandPrixId'].head(1).values[0]
    
    weather_with_grandprix = weatherData[weatherData['grandPrixId'] == next_race_id]

    st.subheader(f"Weather Data for {weather_with_grandprix['fullName'].head(1).values[0]}:")
    st.write(f"Total number of weather records: {len(weather_with_grandprix)}")

    st.dataframe(weather_with_grandprix, column_config=weather_columns_to_display, hide_index=True)

    st.subheader("Past Results:")
    detailsOfNextRace = data[data['grandPrixRaceId'] == next_race_id]

    # Sort detailsOfNextRace by grandPrixYear descending and resultsFinalPositionNumber ascending
    detailsOfNextRace = detailsOfNextRace.sort_values(by=['grandPrixYear', 'resultsFinalPositionNumber'], ascending=[False, True])

    st.write(f"Total number of results: {len(detailsOfNextRace)}")
    st.dataframe(detailsOfNextRace, column_config=columns_to_display, hide_index=True, width=800, height=600)

    ### add groupby stats for average place, average improvement, DNFs, etc.
    #races_and_weather_grouped = races_and_weather.groupby
    # Group by without 'grandPrixYear'
    individual_race_grouped = detailsOfNextRace.groupby(['resultsDriverName']).agg(
        average_starting_position=('resultsStartingGridPositionNumber', 'mean'),
        average_ending_position=('resultsFinalPositionNumber', 'mean'),
        average_positions_gained=('positionsGained', 'mean'),
        driver_races=('resultsFinalPositionNumber', 'count')
    ).reset_index()

    individual_race_grouped = individual_race_grouped.sort_values(by=['average_ending_position'], ascending=[True])

    # Display the grouped data without index
    st.dataframe(individual_race_grouped, hide_index=True, width=800, height=600)

if st.checkbox('Show Raw Data'):

    st.write(f"Total number of results: {len(data)}")

    st.dataframe(data, column_config=columns_to_display,
        hide_index=True,  width=800, height=600)