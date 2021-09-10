# Top 5 Zip Codes in Florida to Invest In
**Author:** Aziza Gulyamova
***
 
## Overview

For this project, I will be acting as a consultant for a real-estate investment firm. The main goal for my project is to identify **Top 5 zip codes** that are worth investing in anfd **forecast the price**. The assumptions that real investment firm has to meet are following: 

* The minimum capital of investment is \$200,000
* The time horizon for investment is 7-10 years
* The goal is to invest in highly urbanized U.S. areas
* Low risk factor (low volatility threshold)
 
***

### Why Florida State?

insert article
***

## Data

The **Home Listing and Sales** dataset was provided by **Zillow Research**. The data represents **median monthly housing sales prices** over the period of April 1996 through April 2018.
 
Each row represents a unique zip code. Each record contains location info and median housing sales prices for each month.
 
The raw CSV contained the following **columns:**

RegionID  
RegionName -- ZIP Code  
City  
State  
Metro   
CountyName  
SizeRank -- rank of the zipcode according to urbanization.

**Link to dataset:** https://www.zillow.com/research/data/

## Plan of Analysis

<details><summary><b>Data Exploration</b></summary>
    <ul>
        <li>Import Packages</li>
        <li>Upload Dataset</li>
        <li>Explore Dataset</li>
        <li>Visual Exploration</li>
    </ul>
</details>     
<details><summary><b>Exploration Analysis</b></summary>
    <ul>
        <li>Minimum Price</li>
        <li>Volatility</li>
        <li>ROI for 24 Months</li>
    </ul>
</details>     
<details><summary><b>Data Preparation</b></summary>
    <ul>
        <li>Sationarity</li>
        <li>Non Stationary to Stationary</li>
        <li>ACF and PACF</li>
    </ul>
</details>     
  
<details><summary><b>BASELINE ARIMA MODEL</b></summary>
    <ul>
        <li><b>Train Test Split</b></li>
        <li>Model Evaluation</li>
        <li>Grid Search for Best C - Value</li>
        <li>Model Summary</li>
    </ul>
</details>     
<details><summary><b>K Nearest Neighbors Classifier</b></summary>
    <ul>
        <li><b>Model 2: KNN with All Features</b></li>
        <li>Model Evaluation</li>
        <li>Model Summary</li>
    </ul>
</details>  
<details><summary><b>Decision Tree Classifier</b></summary>
    <ul>
        <li>Model Evaluation</li>
        <li>Model Summary</li>
    </ul>
</details> 

<b>Modeling Conclusion</b><br>
<b>Evaluation of Final Model</b><br>
<b>Recommendations Based on Final Model</b><br>
<b>Next Step</b>

# Data Exploration
Before proceeding to any analysis and modeling, I will need to upload necessary packages and upload dataset. After that, the data needs to bee explored and cleaned from unnecessary columns and observations.



## Import Packages

Import necessary packages and libraries for data cleaning and manipulation.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
```

from functools import reduce
from datetime import datetime

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

# Import additional files with statistical functions
import sys
import os

module_path = os.path.abspath(os.path.join('./src'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import explore_data as ed
import ts_functions as tsf


# Import and load modeling and forecasting tools

# In[2]:


get_ipython().system('pip install pmdarima')


# In[3]:


from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots

from pmdarima import auto_arima                              # for determining ARIMA orders
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

import itertools


# In[4]:


from statsmodels.tsa.arima.model import ARIMA


# Adjust the notebook settings

# In[5]:


pd.options.display.max_rows = 700
pd.options.display.max_columns = 100
plt.style.use('seaborn')


# ## Upload Dataset

# The dataset is provided by **Zillow Resarch** and stored in **Dataset folder** of the project. I will upload data into **zillow** variable and store all features as an object data type in order to keep possible leading zeros.

# In[6]:


zillow = pd.read_csv('zillow_data.csv', dtype = 'object' )


# In[7]:


zillow.head()


# In[8]:


zillow.info()


# * The data appears to be stored in **wide format**, meaning that the **observations of time feature are stored as individual columns** with median **house price value as observations**. 

# ## Explore Data

# * Extract data for **Florida State**
# * Choose **highly urbanized** zipcodes
# * Remove **unnecessary variables**
# * Check for missing values

# For my project, I chose to focus on **zipcodes of the FL state**, since it is one of the states with **high inbound migration** percentage and also has **no state income taxation**. Thus, the real estate market there is at **high performance.** 

# First, I will select observations only for FL state.

# In[9]:


fl_df = zillow[zillow['State']=='FL']
fl_df.head()


# * Now, I will drop columns that are not needed for my analysis, rename the **RegionName** feature to **Zipcode** and reset index.

# In[10]:


fl_df.drop(columns = ["RegionID", "City", "State", "Metro", "CountyName"], axis=1, inplace = True)
fl_df.rename(columns={'RegionName': 'Zipcode'}, inplace=True)
fl_df.head()


# In[11]:


fl_df.reset_index(drop=True, inplace=True)
fl_df.head()


# ### Highly Urbanized Zip Codes

# I will sort out the zipcodes with highest **SizeRank**, since the real estate market is higly growing where the urbanization is high. I will keep zip codes that are at **top 15 quantile** according to the SizeRank variable.

# In[12]:


#first, convert SizeRank variable to numeric data type
fl_df["SizeRank"]=fl_df["SizeRank"].astype(int)

#calculate the 0.15 quantile
sr_15q = fl_df.SizeRank.quantile(q=0.15) 
# select data only in top 15 quantile of SizeRank
fl_top15= fl_df[fl_df['SizeRank']<sr_15q]


# In[13]:


fl_top15.drop(columns = 'SizeRank', inplace = True, axis = 1)
fl_top15.head()


# In[14]:


# set Zipcode variable as index
fl_top15.set_index('Zipcode', inplace = True)
fl_top15.head()


# In[15]:


# set all features as int data types
fl_top15 = fl_top15.astype(int)

# check for missing values
fl_top15.isna().sum()


# In[16]:


fl_df=fl_top15.transpose() # switch indecees and column names
fl_df.head()


# * Now, I will change the dates to **datetime data type** in order to proceed with Visual EDA and future modeling.

# In[17]:


fl_df.index=pd.to_datetime(fl_df.index, infer_datetime_format=True)
fl_df.info()


# In[18]:


fl_df.head()


# ## Visual Explorations

# To check visually if data has **trends** and **seasonality**, I will plot graph of **prices for each zipcode.**

# In[19]:


fl_df.plot(figsize=(20,15))
plt.title("Housing Price Trends ")
plt.xlabel('Year')
plt.ylabel('Home Price $')
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol= 10, fontsize = 'x-large')
plt.gcf().autofmt_xdate()


# * It is clear that most of the zipcodes has low volatility, but there are some that have noticable trendiness. The downfall in period from 2007 - 2012 had occured due to economic crash in Great Recession period. According to IG.Ellen, one of the truly distinctive features of Great Recession is the severe housing crisis layered on top of all the labor market problems (G.Ellen, 2012).  
# After 2012 the market started to recover. 

# ### Select after 2012

# In[20]:


#fl_df=fl_df.transpose()
#fl_df.head()


# In[21]:


#fl_df.columns.get_loc('2012-01-01')


# In[22]:


#fl_df.drop(fl_df.columns[:189], axis=1,inplace=True)
#fl_df.head()


# In[23]:


#fl_df=fl_df.transpose()
#fl_df.head()


# 

# # Exploration Analysis

# ## Minimum Capital

# ### Medium Price as Investment Capital

# * As per my project assumptions, I will calculate the **minimum amount of capital** that investors suppose to own. In order to determine that minimum amount, I will calculate the **average house prices for 2018 (4 months) and use the median price as a required capital.**

# In[24]:


fl_df = fl_df.transpose()
fl_df.head()


# In[25]:


fl_df.info()


# In[26]:


avg_price = (fl_df['2018-01-01'] + fl_df['2018-02-01'] + 
             fl_df['2018-03-01'] + fl_df['2018-04-01']) / 4
avg_price.head()


# In[27]:


fig1, ax1 = plt.subplots()
ax1.set_title('Median House Price for Zipcodes in 2018')
ax1.boxplot(avg_price)


# * The boxplot shows that at some zipcodes **the median price for the house is extremly high**. I will drop those zipcodes from the dataset, since they **represent outliers.**

# In[28]:


zipcodes = avg_price.index[avg_price > 500000].to_list()
zipcodes


# In[29]:


fl_df.drop(zipcodes, inplace=True)
fl_df.info()


# In[30]:


avg_price.drop(zipcodes, inplace=True)


# In[31]:


fig1, ax1 = plt.subplots()
ax1.set_title('Median House Price for Zipcodes in 2018')
ax1.boxplot(avg_price)


# * The median house price for 4 months appears to be **\\$250,000.** To lower the threshold of entering the market, I will use **$200,000 as a minimum capital to invest.**
# I will drop all zipcodes that have median price lower than \\$200,000

# In[32]:


zipcodes = avg_price.index[avg_price < 200000].to_list()
fl_df.drop(zipcodes, inplace=True)
fl_df.info()


# In[33]:


fl_df.head()


# ## Volatility

# ### Explore the Volatility of the Market

# * To filter out the **zipcodes with average risk for investment**, I will **calculate the volatility of prices at each zipcode**. Goal is to extract zipcodes that have averagre volatility, since **low volatility** means **less returns** and **high volatility** might be **too risky.**
# 
# Since **before 2012 the market had a housing bubble** and that fluctuation might affect the volatility results, I will use observations **only after 2012 to investigate the volatility.**

# In[34]:


fl_df.columns.get_loc('2012-01-01')


# In[35]:


fl_df_2012 = fl_df.drop(fl_df.columns[:189], axis=1)
fl_df_2012.head()


# In[36]:


fl_df_2012=fl_df_2012.transpose()
fl_df_2012.head()


# * To calculate Historical Volatility, I will **use standart deviation of log returns**. First, I will need to **calculate historical ROI for each zipcode.**

# In[37]:


def log_roi (df):
    
    hist_df = pd.DataFrame(columns = df.columns)
    
    for col in df.columns:
        hist_df[col] = (np.log(df[col]/df[col].shift(-1)))
    return hist_df
        
        


# In[38]:


log_roi_1 = log_roi(fl_df_2012)


# In[39]:


log_roi_1


# In[40]:


def std_roi(df):
    
    std_df = pd.DataFrame(columns = ['std'], index = df.columns)
    
    for col in df.columns:
        std_df.loc[[col],['std']] = np.std(df[col])
        #std_df.append(np.std(df[col]))
    return std_df
        


# In[41]:


std_roi_1 = std_roi(log_roi_1)
std_roi_1


# In[42]:


def annual_std(df):
    
    annual_df = pd.DataFrame(columns = ['annual_std', 'volatility'], index = df.index)
    
    for ind in df.index:
        #print(df['Name'][ind], df['Stream'][ind])
        annual_df.loc[[ind],['annual_std']] = df['std'][ind] * 12 **.5
        annual_df.loc[[ind],['volatility']] = round(df['std'][ind] * 12 **.5, 4)*100
    
    
    return annual_df


# In[43]:


annual_s = annual_std(std_roi_1)


# In[44]:


annual_s


# In[45]:


fig1, ax1 = plt.subplots()
ax1.set_title('Median Volatility Rate of Zipcodes')
ax1.boxplot(annual_s['volatility'])


# * The graph shows that the **median value for volatility rate is around 1.7**. I will **keep zipcodes with volatility rate of 1.7-2.9.** based on project assumptions.

# In[46]:


fl_df.info()


# In[47]:


zipcodes = annual_s.index[(annual_s['volatility'] <1.7) | (annual_s['volatility'] >2.8)].to_list()
fl_df.drop(zipcodes, inplace=True)
fl_df.info()


# ## ROI of 24 Months

# ### Calculate the ROI for the last 24 months

# For the purpose of the project, I will calculate the **average 24 months ROI percentage** of each zipcode, in order to find **zipcodes that have highest return.**

# In[48]:


roi = ((fl_df['2018-04-01'] - fl_df['2017-04-01'])/fl_df['2017-04-01']) * 100
roi.sort_values(ascending=False)


# In[49]:


#roi.plot(style='.', figsize=(20,15))
fig1, ax1 = plt.subplots()
ax1.set_title('One Year ROI for Zipcodes')
ax1.boxplot(roi)


# * The **median ROI** for the last 24 months of all zipcodes in Florida is **7%**, but since the project goal is to find best zipcodes to invest in, I will focus on those zipcodes, that have **return rate higher than upper quantile,** which is 9.5%

# In[50]:


zipcodes = roi.index[roi <9.5].to_list()
fl_df.drop(zipcodes, inplace=True)
fl_df.info()


# In[51]:


fl_df = fl_df.transpose()
fl_df.head()


# In[52]:


fl_df.plot(figsize=(15,10))
plt.title("Housing Price Trends ")
plt.xlabel('Year')
plt.ylabel('Home Price $')
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol= 10, fontsize = 'x-large')
#plt.gcf().autofmt_xdate()


# ### Summary:

# The datast had been cleaned out and filtered down as follows:
# * Zipcodes of **Florida State** had been selected
# * Zipcodes with **urbanization rank** in **top 15 quantile**
# * The **minimum median price of housing is \\$200,000** for the last 4 months
# * The **volatility range** of the median price is **1.7 - 2.9**
# * The **ROI rate is greater than 9.5%**
# 
# The initial **amount of zipcodes** was reduced **from 14723 to 14.**

# # Data Preparation

# * Before proceeding to modeling I will check the **data stationarity and autocorrelation of the prices.**

# ## Stationarity of Data

# In[53]:


tsf.stationarity_test(fl_df, 12)


# In[54]:


tsf.dickey_fuller_test(fl_df) # function stored in time_series_funtion file


# * As the **Dickey- Fuller Test** shows, none of the **zipcodes'** data **do not meet stationarity assumption**. I will try different transformation merhods to **modify data before modeling.**

# ## Non Stationary to Stationary

# ### Differencing

# * For now, I will work with 3 zipcodes and try **different methods to transform data to stationary**. After finding most suited method, I will use it on all zipcodes and **check if the stationarity assumption is met.**

# In[55]:


tsf.find_best_difference(fl_df[['33064', '33313', '33319']], 6) # function stored in time_series_funtion file


# * The function above shows p-values of 6 periods of differencing. It is clear that only **33064 zipcode** becomes non-stationary with 1 level of differencing. Next, I will work with **log transformation.**

# ### Log transform

# In[56]:


three_zips = np.log(fl_df[['33064', '33313', '33319']].dropna())


# In[57]:


tsf.dickey_fuller_test(three_zips)


# * Log transformation **didn't** appear to **make the data stationary**, since the critical value assumption is only met by one zipcode.

# ### Differencing logg transformed zipcodes

# In[58]:


tsf.find_best_difference(three_zips.dropna(), 6)


# ### Second order differencing on log transformed data

# In[59]:


tsf.dickey_fuller_test(three_zips.diff().diff().dropna())


# * The **second order differencing** appears to be the **best choice to transform data to stationary**. All three zipcodes have met the critical value assumption, even though the p-values are high.

# ## Transformation of Whole Dataset

# * Now, I will try to use **second order differencing without log transformation** on all dataset.

# In[60]:


tsf.dickey_fuller_test(fl_df.diff().diff().dropna())


#  * **All zipcodes have met the critical value assumption**, but **only 8 out of 12 zipcodes** have **met the p-value parameter**. I will explore the results for log transformed data.

# In[61]:


tsf.dickey_fuller_test(np.log(fl_df).diff().diff().dropna())


# * Clearly, the **best option** for transformation is **second order differencing of data without log transforming** it. For further analysis I will **keep working with & zipcodes that have met critical value assumption and p-value parameter.**

# In[62]:


df_fl = fl_df[['33064', '33157', '32825', '32771', '33463','34698', '33020','33033']].diff().diff()
df_fl.dropna(inplace = True)
df_fl


# In[63]:


df_fl.plot(figsize=(15,10))


# In[64]:


df_fl.plot(figsize = (20,15), subplots=True, legend=True)
plt.show()


# In[65]:


tsf.stationarity_test(df_fl, 12)


# **Final Dataset for Modeling**

# In[66]:


df_final = df_fl.copy()
df_final.head()


# ## ACF and PACF

# In[67]:


#plt.figure(figsize=(12,6))

for col in df_final.columns:
    title = 'Autocorrelation: '+ col + ' Zipcode'
    lags = 40
    plot_acf(df_final[col],title=title,lags=lags)
    #plt.figure(figsize=(15,3))
    #pd.plotting.autocorrelation_plot(df_final[col]);


# The graphs show that the **median prices** of houses are **significantly correlated for first lag** for most of the zipcodes and I can notice that all zipcodes have **negative correlation at different lags.**

# In[68]:


for col in df_final.columns:
    title = 'Partial Autocorrelation:: '+ col + ' Zipcode'
    lags = 40
    plot_pacf(df_final[col],title=title,lags=lags)


# # Baseline ARIMA Model

# ## Train Test Split

# I will segment the **test/forecasting data to only the most recent 3 years.**

# In[69]:


df_final.head()


# * To proceed with modeling, I will create **8 separate datasets** with **time as index and median price as value.**

# In[70]:


df_33064 = pd.DataFrame(df_final['33064'])
df_33064.index.name = 'time'
df_33157 = pd.DataFrame(df_final['33157'])
df_33157.index.name = 'time'
df_32825 = pd.DataFrame(df_final['32825'])
df_32825.index.name = 'time'
df_33463 = pd.DataFrame(df_final['33463'])
df_33463.index.name = 'time'
df_34698 = pd.DataFrame(df_final['34698'])
df_34698.index.name = 'time'
df_33020 = pd.DataFrame(df_final['33020'])
df_33020.index.name = 'time'
df_33033 = pd.DataFrame(df_final['33033'])
df_33033.index.name = 'time'
df_32771 = pd.DataFrame(df_final['32771'])
df_32771.index.name = 'time'


# * The holdout set will be 20% of the data, the test set will consist of 20% of 80% that was left for trainig.

# In[71]:


train_1,test_1,hold_1 = ed.train_test_holdout_split(df_33064.dropna())
train_2,test_2,hold_2 = ed.train_test_holdout_split(df_33157.dropna())
train_3,test_3,hold_3 = ed.train_test_holdout_split(df_32825.dropna())
train_4,test_4,hold_4 = ed.train_test_holdout_split(df_33463.dropna())
train_5,test_5,hold_5 = ed.train_test_holdout_split(df_34698.dropna())
train_6,test_6,hold_6 = ed.train_test_holdout_split(df_33020.dropna())
train_7,test_7,hold_7 = ed.train_test_holdout_split(df_33033.dropna())
train_8,test_8,hold_8 = ed.train_test_holdout_split(df_32771.dropna())


# # Auto - ARIMA for all Zipcodes

# ## 33064 Zipcode Model

# In[72]:


model_1 = auto_arima(train_1,seasonal=True)
tsf.evaluate_auto_arima(model_1, train_1, test_1)


# ## 33157 Zipcode Model

# In[73]:


model_2 = auto_arima(train_2,seasonal=True)
tsf.evaluate_auto_arima(model_2, train_2, test_2)


# ## 32825 Zipcode Model

# In[74]:


model_3 = auto_arima(train_3,seasonal=True)
tsf.evaluate_auto_arima(model_3, train_3, test_3)


# In[75]:


model_4 = auto_arima(train_4,seasonal=True)
tsf.evaluate_auto_arima(model_4, train_4, test_4)


# In[76]:


model_5 = auto_arima(train_5,seasonal=True)
tsf.evaluate_auto_arima(model_5, train_5, test_5)


# In[77]:


model_6 = auto_arima(train_6,seasonal=True)
tsf.evaluate_auto_arima(model_6, train_6, test_6)


# In[78]:


model_7 = auto_arima(train_7,seasonal=True)
tsf.evaluate_auto_arima(model_7, train_7, test_7)


# In[79]:


model_8 = auto_arima(train_8,seasonal=True)
tsf.evaluate_auto_arima(model_8, train_8, test_8)


# * **Model 1** showed the best results in terms of **AIC value (2611)** and **Model 7** showed best results for **Testing RMSE (379.3)**. I will **plot the diagnostics** for those two models to obtain more details.

# In[80]:


model_1.plot_diagnostics(figsize=(7, 10))


# In[81]:


model_7.plot_diagnostics(figsize=(7, 10))


# ### Summary:

# * Standardized residual graphs show that the residual errors fluctuate around 150 for both of the models. 
# * Histogram of the Model 1 shows normally distributed residuals, since KDE and N are close to each other. 
# * The Normal Q-Q graph shows better performance of Model 1
# * Correlogram shows non significant correlation for both models. 

# # Facebook Prophet Model

# I will build a Facebook Prophet Model since it suppose to performe better with seasonal data, be robust to shifts in trend and hande outliers well. 

# The data for FB Prophet should be non differenced, thus I will create new train and test splits for modeling.

# In[82]:


prophet_df = fl_df[['33064', '33157', '32825', '32771', '33463','34698', '33020','33033']]


# In[83]:


pdf_33064 = pd.DataFrame(prophet_df['33064'])
pdf_33157 = pd.DataFrame(prophet_df['33157'])
pdf_32825 = pd.DataFrame(prophet_df['32825'])
pdf_33463 = pd.DataFrame(prophet_df['33463'])
pdf_34698 = pd.DataFrame(prophet_df['34698'])
pdf_33020 = pd.DataFrame(prophet_df['33020'])
pdf_33033 = pd.DataFrame(prophet_df['33033'])
pdf_32771 = pd.DataFrame(prophet_df['32771'])


# In[84]:


ptrain_1,ptest_1,phold_1 = ed.train_test_holdout_split(pdf_33064)
ptrain_2,ptest_2,phold_2 = ed.train_test_holdout_split(pdf_33157)
ptrain_3,ptest_3,phold_3 = ed.train_test_holdout_split(pdf_32825)
ptrain_4,ptest_4,phold_4 = ed.train_test_holdout_split(pdf_33463)
ptrain_5,ptest_5,phold_5 = ed.train_test_holdout_split(pdf_34698)
ptrain_6,ptest_6,phold_6 = ed.train_test_holdout_split(pdf_33020)
ptrain_7,ptest_7,phold_7 = ed.train_test_holdout_split(pdf_33033)
ptrain_8,ptest_8,phold_8 = ed.train_test_holdout_split(pdf_32771)


# In[85]:


# the function returns the datasets with resetted index and column names ds and y, 
# since that's a FB Prophet convention
ed.prophet_df([ptrain_1, ptest_1])
ed.prophet_df([ptrain_2, ptest_2])
ed.prophet_df([ptrain_3, ptest_3])
ed.prophet_df([ptrain_4, ptest_4])
ed.prophet_df([ptrain_5, ptest_5])
ed.prophet_df([ptrain_6, ptest_6])
ed.prophet_df([ptrain_7, ptest_7])
ed.prophet_df([ptrain_8, ptest_8])


# In[86]:


get_ipython().system('pip install pystan==2.17.1.0')


# In[87]:


get_ipython().system('pip install fbprophet')


# In[88]:


from fbprophet import Prophet


# In[89]:


pmodel_1 = Prophet(weekly_seasonality=True, daily_seasonality=True)
pmodel_1.fit(ptrain_1)


# In[90]:


pdf_1 = pmodel_1.make_future_dataframe(periods=len(ptest_1), freq='MS')


# In[91]:


forecast_1 = pmodel_1.predict(pdf_1)
forecast_1.head()


# In[92]:


pmodel_1.plot(forecast_1, uncertainty=True)
plt.show()


# In[93]:


pmodel_1.plot_components(forecast_1)
plt.show()


# * Now I will plot forecast values with actual values and calculate the RMSE

# In[97]:


y_pred = forecast_1['yhat'][-22:].values

testing_MSE = mean_squared_error(ptest_1, y_pred)**.5
print('Testing RMSE = ', testing_MSE)

#mae_1 = mean_absolute_error(y_true, y_pred)
#print('MAE: %.3f' % mae)

# plot expected vs actual
plt.plot(ptest_1.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()


# In[98]:


pmodel_7 = Prophet(weekly_seasonality=True, daily_seasonality=True)
pmodel_7.fit(ptrain_7)


# In[99]:


pdf_7 = pmodel_7.make_future_dataframe(periods=len(ptest_7), freq='MS')


# In[100]:


forecast_7 = pmodel_7.predict(pdf_7)
forecast_7.head()


# In[101]:


pmodel_7.plot(forecast_7, uncertainty=True)
plt.show()


# In[102]:


pmodel_7.plot_components(forecast_7)
plt.show()


# * Now I will plot forecast values with actual values and calculate the RMSE

# In[103]:


y_pred = forecast_7['yhat'][-22:].values

testing_MSE = mean_squared_error(ptest_7, y_pred)**.5
print('Testing RMSE = ', testing_MSE)

#mae_1 = mean_absolute_error(y_true, y_pred)
#print('MAE: %.3f' % mae)

# plot expected vs actual
plt.plot(ptest_7.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()


# In[104]:


pmodel_2 = Prophet(weekly_seasonality=True, daily_seasonality=True)
pmodel_2.fit(ptrain_2)


# In[105]:


pdf_2 = pmodel_2.make_future_dataframe(periods=len(ptest_2), freq='MS')


# In[106]:


forecast_2 = pmodel_2.predict(pdf_2)
forecast_2.head()


# In[107]:


pmodel_2.plot(forecast_2, uncertainty=True)
plt.show()


# In[108]:


pmodel_2.plot_components(forecast_2)
plt.show()


# * Now I will plot forecast values with actual values and calculate the RMSE

# In[109]:


y_pred = forecast_2['yhat'][-22:].values

testing_MSE = mean_squared_error(ptest_2, y_pred)**.5
print('Testing RMSE = ', testing_MSE)

#mae_1 = mean_absolute_error(y_true, y_pred)
#print('MAE: %.3f' % mae)

# plot expected vs actual
plt.plot(ptest_2.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()


# ### Summary:

# After building models for three datasets, it is clear that Facebook Prophet performed worse than ARIMA models. The Mean Squared Error is higher with Prophet and due to noticable fluctuation in dataset, the model forecasts in downward direction, whereas the actual values go up.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[96]:


model_2 = gARIMA(train_1,order=(2,0,0))
results = model_2.fit()
results.summary()


# In[ ]:


train_df = [train_1, train_2, train_3, train_4, train_5, train_6, train_7]
test_df = [test_1, test_2, test_3, test_4, test_5, test_6, test_7]
names = ['33064', '33157', '32825', '33463','34698', '33020','33033']


# In[ ]:


# Establish a metrics dataframe to store models
column_metrics = ['Name', 'Order', 'Seasonal_Order', 
                  'Const', 'ar.L1', 'ma.L1', 'sigma2', 'AIC Score']
metrics_df = pd.DataFrame(columns = column_metrics)

# Start with a baseline model of AR(1) and MA(1)
order=(1,0,1)

for i, t_df in enumerate(train_df):
    metrics_df = tsf.run_arima_models(names[i], t_df, test_df[i], order, metrics_df) # Function in user_functions.py

metrics_df


# In[ ]:


# Model with order (1,0,0)
order=(1,0,0)

for i, t_df in enumerate(train_df):
    metrics_df = tsf.run_arima_models(names[i], t_df, test_df[i], order, metrics_df) # Function in user_functions.py

metrics_df
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


ar1 = ARIMA(train_1, order=(1,0,0)).fit()
tsf.evaluate_model(ar1, df_33064, train_1, test_1)


# In[ ]:


test_1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


model_1 = ARIMA(train_1, order=(1,0,0)).fit()
tsf.evaluate_model(model_1, df_33064, train_1, test_1)


# In[ ]:


dff_33064.index.name = 'time'
df_33064


# In[ ]:


train_1,test_1,hold_1 = ed.train_test_holdout_split(df_33064)


# In[ ]:


len(test_split['33064'])


# In[ ]:


trainpreds = model_1.forecast(12)
type(trainpreds)


# In[ ]:


type(train_1)


# In[ ]:


trainpreds[0]


# In[ ]:


train_set.index.to_list()


# In[ ]:


train_set.index
train_sp = pd.DataFrame(index = train_set.columns, columns =train_set.index.to_list())
train_sp['33064'] = train_split['33064']
train_sp


# In[ ]:


train_sp['33064']


# In[ ]:


df_f


# In[ ]:




