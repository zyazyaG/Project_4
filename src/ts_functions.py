#!/usr/bin/env python
# coding: utf-8

# # Time_Series_Functions

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def stationarity_test(data, window):
    
    #Rolling statistics
    r_mean = data.rolling(window=window).mean()
    r_std = data.rolling(window=window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(15, 10))
    orig = plt.plot(data.iloc[window:], color='green',label='Original')
    mean = plt.plot(r_mean, color='red', label='Rolling Mean')
    std = plt.plot(r_std, color='blue', label = 'Rolling Std')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol= 5, fontsize = 'large')
    plt.title('Rolling Mean & Standard Deviation of Prices')
    plt.show()
    


# In[ ]:


def dickey_fuller_test(data):
    """
    A function to apply a Dicky Fuller test and return the results in a readable dictionary
    """
    full_dict = list()
    
    for col in data.columns:
        dicky_fuller = dict()
        dicky_fuller['____'] = '___________________________'
        dicky_fuller['zipcode'] = col
        
        results = adfuller(data[col])
        keys = ['test statistic','pvalue','usedlag','number of observations','critical values','best information criterion']
        values = results
        for key, value in zip(keys, values):
            dicky_fuller[key] = value
        full_dict.append(dicky_fuller)
    return full_dict

def find_best_difference(data, ranges):
    for col in data.columns:
        for i in range(0,ranges):
            difference = data[col].diff(periods=i)
            dftest = adfuller(difference[i:])
            print(i,dftest[1])
            
            
            
def sarimax_model(trains, tests, metrics_df, train_predict_start=0):
    """
    Takes a fitted ARIMA model, the original data, the train split and the test split.
    optionally, can take a training starting point between 0 and len(train).  default = 0
    optionally, can take forecast length to determine how far to forecast
    shows a chart of data, train prediction, and test prediction
    prints the MSE of the training prediction and the testing prediction
    returns a summary object from model.summary()
    """
    
    for i in range(len(trains)):
        model = SARIMAX(trains[1],order=(2,0,0)).fit()
    
        trainpreds = model.predict()
        testpreds = model.forecast(len(tests[i]))
    

        training_MSE = mean_squared_error(trains[i], trainpreds)**.5
        testing_MSE = mean_squared_error(tests[i], testpreds)**.5
        
        
        metrics_df.loc[[list(trains[i].columns)[0]],['AIC']] = round(model.aic, 2)
        metrics_df.loc[[list(trains[i].columns)[0]],['Training RMSE']] = training_MSE
        metrics_df.loc[[list(trains[i].columns)[0]],['Testing RMSE']] = testing_MSE
        
        #plot the predictions for test set
        plt.plot(trains[i], label='Train')
        plt.plot(tests[i], label='Test')
        plt.plot(testpreds, label='Prediction')
        plt.legend()
        plt.title(f"{list(trains[i].columns)[0]} Zipcode")
        plt.show()
        
    
    
    return 

def sarimax_final_model(trains, tests, holds, metrics_df, train_predict_start=0):
    """
    Takes a fitted ARIMA model, the original data, the train split and the test split.
    optionally, can take a training starting point between 0 and len(train).  default = 0
    optionally, can take forecast length to determine how far to forecast
    shows a chart of data, train prediction, and test prediction
    prints the MSE of the training prediction and the testing prediction
    returns a summary object from model.summary()
    """
    forecast_df = []
    
    for i in range(len(trains)):
        model = SARIMAX(trains[1],order=(2,0,0)).fit()
    
        trainpreds = model.predict()
        testpreds = model.forecast(len(holds[i])+ len(tests[i]))
        df = pd.DataFrame(testpreds)
        forecast_df.append(df)
        

        training_MSE = mean_squared_error(trains[i], trainpreds)**.5
        testing_MSE = mean_squared_error(pd.concat([tests[i],holds[i]]), testpreds)**.5
        
        
        metrics_df.loc[[list(trains[i].columns)[0]],['AIC']] = round(model.aic, 2)
        metrics_df.loc[[list(trains[i].columns)[0]],['Training RMSE']] = training_MSE
        metrics_df.loc[[list(trains[i].columns)[0]],['Testing RMSE']] = testing_MSE
        
        #plot the predictions for test set
        plt.plot(trains[i], label='Train')
        plt.plot(tests[i], label='Test')
        plt.plot(holds[i], label='Holdout')
        plt.plot(testpreds, label='Prediction')
        plt.legend()
        plt.title(f"{list(trains[i].columns)[0]} Zipcode")
        plt.show()
        
    return  forecast_df

def run_arima_models(name, train, test, order, metrics_df, seasonal_order = (0,0,0,0)):
    """Runs baseline ARIMA model and adds metrics and results to a passed dataframe"""
    
    model_metrics = [name, order, seasonal_order]
    
    model = ARIMA(train, order=order, seasonal_order=seasonal_order, freq='MS')
    results = model.fit()
    
    
  
    
    #Print out summary information on the fit
    #print(results.summary())
    
    model_metrics.extend([round(results.params[0], 2), round(results.params[1], 4), 
                          round(results.params[2], 4)])
    if len(results.params) > 3:
        model_metrics.append([round(results.params[3], 4)])
    else:
        model_metrics['ma.L1'] = 0
    
    model_metrics.append(round(results.aic, 2))
    
    
    # Get predictions starting from first test index and calculate confidence intervals
    
    # pred = results.get_prediction(start = test.index[0], end = test.index[-1], dynamic=True, full_results=True)
    # pred_conf = pred.conf_int()
  
    
    # Add model metrics to passed metrics df    
    series = pd.Series(model_metrics, index = metrics_df.columns)
    metrics_df = metrics_df.append(series, ignore_index=True)
    
    return metrics_df

def evaluate_auto_arima(model, train, test, metrics_df):
    
    forecast = model.predict(n_periods=len(test))
    forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])

    #plot the predictions for test set
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(forecast, label='Prediction')
    plt.title(f"{list(train.columns)[0]} Zipcode")
    plt.legend()
    plt.show()
    
    testing_MSE = mean_squared_error(test, forecast)**.5
    print('Testing RMSE = ', testing_MSE)
    
    metrics_df.loc[[list(train.columns)[0]],['Order']] = str(list(model.get_params().get('order')))
    metrics_df.loc[[list(train.columns)[0]],['AIC']] = round(model.aic(), 2)
    metrics_df.loc[[list(train.columns)[0]],['Testing RMSE']] = testing_MSE
    
    #params.extend([round(forecast.params[0], 2), round(results.params[1], 4), 
                          #round(results.params[2], 4)])
  
    #params.append(round(forecast.aic, 2))
    # Add model metrics to passed metrics df    
    #df = pd.Series(params, index = metrics_df.columns)
    
    #metrics_df = metrics_df.append(series, ignore_index=True)
    
    return   model.summary()