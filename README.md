# Formula 1 Data Analysis
Analysis of Formula 1 ```.json``` files based on the very generous data files from [F1DB](https://github.com/f1db/f1db) for the vast majority of the analysis. F1DB did not have race control messages which include Safety Cars and flags. For that data, I used [FastF1](https://docs.fastf1.dev/). Full data analysis is available through the [Formula 1 Analysis - Streamlit app](https://f1analysis-app.streamlit.app/).

## File organization
There are two python files involved in this app: ```raceAnalysis.py``` and ```f1-generate-analysis.py```, though there are other python files which generate content. The Race Analysis file is what runs the Streamlit code and displays the data, filters, charts, etc. Before that file is run, you need to run the Generate Analysis page. This creates a bunch of dataframes, and it creates several .csv files for easier retrievel during the Streamlit display. This is done so fewer calculations are required in the Streamlit app which should improve performance. However, it does require that you run the ```f1-generate-analysis.py``` before you run the Steamlit.

The CSV files and any associated .json files are included in the ```data_files``` directory. The ```.json``` files come from [F1DB](https://github.com/f1db/f1db). The following files are generated and then copied into the ```data_files``` directory:

1. ```f1WeatherData_Grouped.csv```
2. ```f1PitStopsData_Grouped.csv```
3. ```f1ForAnalysis.csv```
4. ```f1db-races.json```
5. ```f1db-drivers.json```
6. ```f1db-grands-prix.json```
7. ```f1db-races-race-results.json```

## Filtering
There are currently more than 30 ways to filter the F1 data which spans from 2015 to present. You can filter by one or all of the data fields on the left side of the page. The data dynamically updates and gives you a new total record count. 

## Linear regression
In addition to correlation coefficients, I have added several linear regressions to help predict the results of the next race. 

## Predictive Data Modeling
I used [sckit-learn](https://scikit-learn.org/stable/) to perform machine learning by using data points to predict the race winner. ~~The model is in its infancy, and I am still trying to figure out the right data points to feed it.~~ I'm also currently trying to predict a driver's final place rather than their final time. That means that the [Mean Absolute Error](https://www.sciencedirect.com/topics/engineering/mean-absolute-error) relates to finisher placement which feels less exact than what I need. I'm using the XGBoost model. The predictive modeling is now under Advanced Options.

I have added [Monte Carlo](https://www.ibm.com/think/topics/monte-carlo-simulation), [Recursive Feature Elimination (RFE)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html), and [Boruta](https://www.jstatsoft.org/v36/i11/) feature selection to pair down the data fields. After significant refinement, I have a MAE down to 1.7 or less.

## Other options
Besides filtering, you can also look at the upcoming race which shows historical and upcoming weather, the past winners, and data about the constructors. You can view the entire current season with details about each file. You can look at the raw, unfiltered data. Finally, you can view a correlation for the entire dataset.

## Weather
The weather is pulled from [Open-Meteo's free API](https://open-meteo.com/) which allows you to search historical weather data by hour going back to the 1940s. The hourly reports are pulled per race and then averaged to show a daily weather report on race day.


### To do
- Figure out a way to reset the filters.
- ~~Incorporate the linear regression equations for predictive race results.~~
