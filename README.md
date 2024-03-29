## Bike rental demand prediction
<b>Introduction</b>

This notebook summarize a 4-day machine learing datathon (or hackathon) I participated in as a student at Henry Academy's data science bootcamp.(https://www.soyhenry.com/carrera-data-science)

<b>The problem</b>:

"We" are part of the Rent-Cycle team in Washington DC, and our Team Leader gives us the task of implementing a model that allows us to predict the number of bicycles that are rented based on the information contained in the dataset made available.
We need to use the Root Mean Square Error (RMSE) as a metric to evaluate our model!

## Project overview
* Data exploration and analysis of the variables that can effectively predict the number of bicycles 
* Feature Binning to clear the noise of data and prevent overfitting
* Optimized Decision Tree, Random Forest and Ridge Regressors using GridsearchCV to reach the best model. 


## Code and Resources Used 
**Python Version:** 3.10.4  
**Packages:** pandas, numpy, pandas_profiling, sklearn, matplotlib, seaborn  
**Data science project from scratch:** https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t  


## Dataset dimensions provided:
The dataset already had feature engineering work, we got the following:

* instant: record index
* dteday : date
* season : season (1:springer, 2:summer, 3:fall, 4:winter)
* yr : year (0: 2011, 1:2012)
* mnth : month ( 1 to 12)
* hr : hour (0 to 23)
* holiday : weather day is holiday or not (extracted from [Web Link])
* weekday : day of the week
* workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
* weathersit :

      1: Clear, Few clouds, Partly cloudy, Partly cloudy
      2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
      3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
      4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
* temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
* atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
* hum: Normalized humidity. The values are divided to 100 (max)
* windspeed: Normalized wind speed. The values are divided to 67 (max)
* casual: count of casual users
* registered: count of registered users
* cnt: count of total rental bikes including both casual and registered

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from EDA. 

![alt text](https://github.com/cvillarragamo/Datathon-Bike-Rental-ML/blob/main/figures/linear_correlation.png "linear correlation")
![alt text](https://github.com/cvillarragamo/Datathon-Bike-Rental-ML/blob/main/figures/distributions.png "Distribution")

## Model Building 

First, I use the columns as provided since they were encoded and standardized. I also split the data into train and tests sets with a test size of 20%.   

I tried three different models thinking that outliers aren’t particularly bad for this type of model.   

*	**Decision Tree** – Baseline for the model
*	**Random Forest** – Because of the sparse data from the many categorical variables, I thought a would be effective.
*	**Ridge Regressor** – Because it uses parameters that "penalize" flexibility in an attempt to minimize overfitting, so I thought that this would be a good fit. 

## Feature binning
After the first-baseline model runned, I needed to adjust some variables to see if it really improved the model. I create the following variables, each one with 4 bins based on rank of quartiles:

* temp_ 
* hum_
* hr_

![alt text](https://github.com/cvillarragamo/Datathon-Bike-Rental-ML/blob/main/figures/featured.png "Feature binning")

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
The feature binning create a overfitting of the model, so the baseline was a better choice.


## Optimization 
I run out of time to optimized the model with advanced regressors and cross validation
