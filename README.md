# Top 5 Zip Codes in Florida to Invest In
**Author:** Aziza Gulyamova
***

![florida](images/florida.png)

## Overview

For this project, I will be acting as a consultant for a real-estate investment firm. The main goal for my project is to identify **Top 5 zipcodes** that are worth investing in and **forecast the price** for future. The assumptions that real investment firm has to meet are following: 

* The minimum capital of investment is \$200,000
* The goal is to invest in highly urbanized U.S. areas
* Low risk factor (low volatility threshold)
 
***

### Why Florida State?

![florida1](images/florida1.png)

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

***

## Analysis Outline

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

## Data Cleaning and Explorations

The dataset is stored in **wide format**, meaning all observations of **time feature** are stored as **individual columns** with median **house price of each zipcode as observations**:

![data](images/data.png)

Since my focus for the project is **zipcodes of Florida State**, I've filtered out needless zipcode observations and selected zipcodes that are in **top 15 quantile by SizeRank** variable. At the end, the dataset contained **only 118 zipcodes out of 14,723.** 

Before proceeding with explorations, I've converted **time feature** variables **into datetime** datatype, **transposed the dataset** and **removed unnecessary features**, such that "RegionID", "City", "State", "Metro", "CountyName".

At the end of the cleaning process, the dataset looked as following:

![data1](images/data1.png)

To check visually if data has **trends and seasonality**, I plotted **graph of median prices for each zipcode**.

![graph1](images/graph1.png)

- It is clear that **most of the zipcodes has low volatility**, but there are some that have **noticable trendiness**. The downfall in period **from 2007 - 2012** had occured due to **economic crash in Great Recession period**. According to IG.Ellen, one of the truly distinctive features of Great Recession is the **severe housing crisis** layered on top of all the labor market problems (G.Ellen, 2012).
After 2012 the market started to recover.
