# Formula 1 Data Analysis
Analysis of Formula 1 ```.json``` files based on the very generous data files from [F1DB](https://github.com/f1db/f1db). Full data analysis is available through the [Formula 1 Analysis - Streamlit app](https://f1analysis-app.streamlit.app/).

## File organization
There are two python files involved in this app: ```raceAnalysis.py``` and ```f1-generate-analysis.py```. The Race Analysis file is what runs the Streamlit code and displays the data, filters, charts, etc. Before that file is run, you need to run the Generate Analysis page. This creates a bunch of dataframes, and it creates several .csv files for easier retrievel during the Streamlit display. This is done so fewer calculations are required in the Streamlit app which should improve performance. However, it does require that you run the ```f1-generate-analysis.py``` before you run the Steamlit.

The CSV files and any associated .json files are included in the ```data_files``` directory. The ```.json``` files come from [F1DB](https://github.com/f1db/f1db). The following files are generated and then copied into the ```data_files``` directory:

1. ```f1WeatherData_Grouped.csv```
2. ```f1PitStopsData_Grouped.csv```
3. ```f1ForAnalysis.csv```
4. ```f1db-races.json```
5. ```f1db-grands-prix.json```

## Filtering
There are currently more than 30 ways to filter the F1 data which spans from 2015 to present. You can filter by one or all of the data fields on the left side of the page. The data dynamically updates and gives you a new total record count. 

## Other options
Besides filtering, you can also look at the upcoming race which shows historical and upcoming weather, the past winners, and data about the constructors. You can view the entire current season with details about each file. You can look at the raw, unfiltered data. Finally, you can view a correlation for the entire dataset.

## Weather
The weather is pulled from [Open-Meteo's free API](https://open-meteo.com/) which allows you to search historical weather data by hour going back to the 1940s. The hourly reports are pulled per race and then averaged to show a daily weather report on race day.


### To do
- Figure out a way to reset the filters.
