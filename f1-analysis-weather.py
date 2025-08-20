import openmeteo_requests
import datetime as dt
import requests_cache
import numpy as np
import pandas as pd
from retry_requests import retry
from openmeteo_sdk.Variable import Variable
from os import path
import os

DATA_DIR = 'data_files/'

current_year = dt.datetime.now().year
raceNoEarlierThan = current_year - 10

races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json')) 
circuits = pd.read_json(path.join(DATA_DIR, 'f1db-circuits.json')) 


races = races[races['year'].between(raceNoEarlierThan, current_year-1)]

circuits_and_races = pd.merge(races, circuits, left_on='circuitId', right_on='id', suffixes=['_races', '_circuits'])
circuits_and_races.columns
circuits_and_races[['id_races', 'circuitId', 'year', 'date', 'grandPrixId', 'latitude', 'longitude']]

circuits_and_races_lat_long = circuits_and_races[['id_races', 'latitude', 'longitude', 'date', 'grandPrixId', 'circuitId']]


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"

all_hourly_data = []

full_params = []

for race in circuits_and_races_lat_long.itertuples():
    params = {
    "latitude": race.latitude,
	"longitude": race.longitude,
	"start_date": race.date.strftime('%Y-%m-%d'),
	"end_date": race.date.strftime('%Y-%m-%d'),
	"hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m"],
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph"
	}

    full_params.append(params)


# Loop through the list of params
for params in full_params:
    responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

# Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
	    start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	    end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	    freq = pd.Timedelta(seconds = hourly.Interval()),
	    inclusive = "left"
)}

    hourly_data["latitude"] = response.Latitude()
    hourly_data["longitude"] = response.Longitude()
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["hourly_precipitation"] = hourly_precipitation
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["short_date"] = pd.to_datetime(hourly_data["date"]).strftime('%Y-%m-%d')
    hourly_dataframe = pd.DataFrame(data = hourly_data)
    
    all_hourly_data = pd.DataFrame(data = all_hourly_data)

    all_hourly_data = pd.concat([all_hourly_data, hourly_dataframe], ignore_index=True)
    
   # date_for_merge = 
circuits_and_races_lat_long['date'] = pd.to_datetime(circuits_and_races_lat_long['date']).dt.strftime('%Y-%m-%d')
races_and_weather = pd.merge(all_hourly_data, circuits_and_races_lat_long, left_on='short_date', right_on='date', how='inner', suffixes=['_hourly', '_lat_long'])

print(races_and_weather)


races_and_weather.to_csv('f1WeatherData_AllData.csv', columns=['date_hourly', 'latitude_hourly', 'longitude_hourly', 'temperature_2m', 'hourly_precipitation', 'relative_humidity_2m', 'short_date',
'wind_speed_10m', 'id_races', 'grandPrixId', 'circuitId'], sep='\t')

races_and_weather_grouped = races_and_weather.groupby(['short_date', 'latitude_hourly', 'longitude_hourly', 'id_races', 'grandPrixId', 'circuitId']).agg(average_temp = ('temperature_2m', 'mean'), total_precipitation = ('hourly_precipitation', 'sum'), average_humidity = ('relative_humidity_2m', 'mean'), average_wind_speed = ('wind_speed_10m', 'mean')).reset_index()

races_and_weather_grouped.to_csv('f1WeatherData_Grouped.csv', columns=['short_date', 'id_races', 'grandPrixId', 'circuitId', 'latitude_hourly', 'longitude_hourly', 'average_temp', 'total_precipitation', 'average_humidity', 'average_wind_speed'], sep='\t')

print(races_and_weather_grouped)