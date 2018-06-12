# -*- coding: utf-8 -*-
"""
Created in 2018

@author: JiaRong
"""

% reset -f


####------------------------------ 1. Settings ------------------------------####
# !pip install cython pystan 
# !pip install fbprophet

# Import python packages
import numpy as np
import pandas as pd
import time
from time import strftime
from datetime import datetime
from datetime import timedelta
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Layout
from plotly.offline import init_notebook_mode, iplot
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric

# Import custom user-defined functions
from stock_data_download import get_stocks_info
from stock_data_download import historical_extract
from stock_calculation import get_log_returns
from stock_calculation import metrics_calculation

# Define variables  
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
duration_year = 5
duration_day = 1
today = datetime.now().date()
end_date = today
start_date = today.replace(year=today.year-duration_year, day = today.day+duration_day)
tradeday_per_year = 252
country_code = 42 # code for KLSE in investing.com



####------------------------------ 2. Data Collection and Web Scrapping ------------------------------####
# Scrap stocks information from Investing.com and i3investor.com
stocks_list = get_stocks_info(country_code, user_agent)
stocks_list.head(5)
print('Statistics about stocks list:\n\n', stocks_list.describe())
print('Unique value in columns of stocks list:\n')
print(stocks_list.count())

# Scrap the historical prices of stocks
historical_data = list(map(lambda x: historical_extract(x,start_date,end_date,user_agent), stocks_list['price_weblink'].values))
historical_data[0]

# store data as respective columns in dataframe
stocks_list['GICS_sector'] = list(map(lambda x: str(x[1]), historical_data))
stocks_list['stock_code'] = list(map(lambda x: str(x[2]), historical_data))
stocks_list['historical_price'] = list(map(lambda x: x[0], historical_data))
stocks_list['GICS_sector'] = stocks_list['GICS_sector'].astype('category')

stocks_list.head(5)

# Save data as pickle file
#stocks_list.to_pickle('stocks_list.pickle')
# Read pickle file
stocks_list = pd.read_pickle('stocks_list.pickle')



####------------------------------ 3. Find out high potential stocks ------------------------------####
# Exploratory Data Analysis
# 1. Bar Chart
ax1 = stocks_list['GICS_sector'].value_counts().sort_values().plot(kind='barh', color='slateblue')
ax1.set_title('Stocks by GICS sector', size=20)
ax1.set_xlabel('Number of stocks', size=14)
ax1.set_ylabel('GICS sector', size=14)
for i in ax1.patches:
    ax1.text(i.get_width()+0.25, i.get_y()+0.1, str(i.get_width()), color='dimgrey', fontweight='bold', fontsize=12)
plt.show()

# 2. Pie Chart
sector_series = pd.DataFrame(stocks_list['GICS_sector'].value_counts())
plotly.offline.plot({
        "data":[go.Pie(values = sector_series['GICS_sector'], labels = sector_series.index, 
                       name = "Bursa Stocks", hole = .6,)],
        "layout":Layout(title="Stocks by GICS sector")
})
    
    
# Calculate the return and risk of stocks
stocks_list['log_returns'] = list(map(lambda x: get_log_returns(x,'daily'), stocks_list['historical_price'].values))
stocks_list['mean_log_returns'] = (stocks_list['log_returns']).apply(lambda x: np.nanmean(x['return_value']))
stocks_list['sd_log_returns'] = (stocks_list['log_returns']).apply(lambda x: np.nanstd(x['return_value']))
stocks_list['trade_days'] = stocks_list['historical_price'].apply(lambda x: len(x['Price']))

cols = stocks_list.columns.tolist()
cols =  (cols[0:3] + cols[85:87]) + (cols[3:85] + cols[87:])
stocks_list = stocks_list[cols]

# Take a look at the summary of mean & standard deviation of log returns
stocks_list['mean_log_returns'].describe()
stocks_list['sd_log_returns'].describe()
# median of mean daily returns
np.exp(-0.000122)
# highest mean daily returns
np.exp(0.007974)

# Set limits of xaxis and yaxis
xlower= 0.005; xupper= 0.04; ylower= -0.003; yupper= 0.008;
# Visualization(mean and std of log returns)
iplot({
    "data": [go.Scatter(x= stocks_list['sd_log_returns'], y= stocks_list['mean_log_returns'], showlegend= True,
                        mode = 'markers', marker=dict(size=(stocks_list['sales_growth_5yr(%)'])*(1/2),
                                                      opacity=0.6,
                                                      color=stocks_list['eps_growth_5yr(%)'], #set color equal to a variable
                                                      colorbar=dict(title='EPS growth(%) over last 5 years', titlefont=dict(size=10)), colorscale='Blue',showscale=True),
                        text = "<em>" + 'Company: ' + stocks_list['name'] + '</em><br>' +
                                '(' + stocks_list['stock_code'] + ')' + '<br>' +
                                list(map(lambda p: str(p), list(map(lambda x: 'GICS sector: {}'.format(x), stocks_list['GICS_sector'])))) + '<br>' +
                                'Industry: ' + stocks_list['industry'] + '<br>' +
                                list(map(lambda p: str(p), list(map(lambda x: 'Last price: {}'.format(x), stocks_list['last'])))) + '<br>' +
                                list(map(lambda p: str(p), list(map(lambda x: 'No. of trade days: {}'.format(x), stocks_list['trade_days'])))) + '<br>' +
                                'Monthly call: ' + stocks_list['summary_monthly'] + '<br>' +
                                'Weekly call: ' + stocks_list['summary_weekly'] + '<br>' +
                                'Daily call: ' + stocks_list['summary_daily']
                                #'Change of Price last 3 years(%): ' + stocks_list['3year_change(%)']
                                #'Earnings Per Share change(TTM): ' + stocks_list['eps_TTM_change_1yr(%)'] + '<br>'
                                #'Net profit margin(%): ' + stocks_list['netprofit_margin_TTM(%)']
                                )],
    "layout": Layout(
                title="Malaysia Bursa Analysis: Stock Risk vs Reward, " + end_date.strftime("%m/%d/%Y"),
                xaxis=dict(title='Risk/Variability (StDev Log Returns)',
                           titlefont=dict(family='Courier New, monospace',size=16,color='black'), range=[xlower, xupper]),
                yaxis=dict(title='Reward/Growth (Mean Log Returns)',
                           titlefont=dict(family='Courier New, monospace',size=16,color='black'), range=[ylower, yupper])
                    )
})    



# Isolate stocks which have an unique combination of high mean and low standard deviation log returns
# Please take sector into consideration.
stocks_list['rank_mean_log_returns'] = stocks_list['mean_log_returns'].rank(ascending=0, method='dense')
stocks_list['rank_bysector_mean_log_returns'] = stocks_list.groupby('GICS_sector')['mean_log_returns'].rank(pct=False, ascending=0, method='dense')

days_difference = (end_date - start_date).days
stocks_tp = stocks_list[(stocks_list['rank_mean_log_returns'] <100) & (stocks_list['sd_log_returns']<np.percentile(stocks_list['sd_log_returns'],75))
                        & (stocks_list['trade_days']>0.5*days_difference)]
essential_indexes = ['name','GICS_sector','industry','stock_code','summary_monthly','summary_weekly','summary_daily','last','one_year_return(%)','market_cap','pe_ratio','revenue','eps','dividend_yield(%)','price_to_sales_TTM',
                     'price_to_cash_flow_MRQ','price_to_book_MRQ','eps_TTM_change_1yr(%)','eps_growth_5yr(%)','sales_TTM_change_1yr(%)','sales_growth_5yr(%)',
                     'netprofit_margin_TTM(%)','netprofit_margin_5yr(%)','dividend_yield_avg_5yr(%)','dividend_growth_rate(%)','historical_price',
                     'log_returns','mean_log_returns','sd_log_returns','trade_days']
stocks_tp = stocks_tp[essential_indexes].reset_index(drop=True)

stocks_tp.head()
# number of potential stocks 
len(stocks_tp)

plotly.offline.plot({
    "data": [go.Scatter(x= stocks_tp['sales_TTM_change_1yr(%)'], y= stocks_tp['mean_log_returns'], showlegend= True,
                        mode = 'markers', marker=dict(size=(stocks_tp['netprofit_margin_TTM(%)']),
                                                      opacity=0.6,
                                                      color=stocks_tp['eps_growth_5yr(%)'], #set color equal to a variable
                                                      colorbar=dict(title='EPS growth(%) over last 5 years', titlefont=dict(size=10)), colorscale='Blue',showscale=True),
                        text = "<em>" + 'Company: ' + stocks_tp['name'] + '</em><br>' +
                                '(' + stocks_tp['stock_code'] + ')' + '<br>' +
                                list(map(lambda p: str(p), list(map(lambda x: 'GICS sector: {}'.format(x), stocks_tp['GICS_sector'])))) + '<br>' +
                                'Industry: ' + stocks_tp['industry'] + '<br>' +
                                list(map(lambda p: str(p), list(map(lambda x: 'Last price: {}'.format(x), stocks_tp['last'])))) + '<br>' +
                                list(map(lambda p: str(p), list(map(lambda x: 'Revenue growth over 5 years(%): {}'.format(x), stocks_tp['sales_growth_5yr(%)'])))) + '<br>' +
                                list(map(lambda p: str(p), list(map(lambda x: 'Profit margin over 5 years(%): {}'.format(x), stocks_tp['netprofit_margin_5yr(%)'])))) + '<br>' +
                                list(map(lambda p: str(p), list(map(lambda x: 'EPS growth last year(%): {}'.format(x), stocks_tp['eps_TTM_change_1yr(%)'])))) + '<br>' +
                                list(map(lambda p: str(p), list(map(lambda x: 'Average dividend yield over 5 years(%): {}'.format(x), stocks_tp['dividend_yield_avg_5yr(%)'])))) + '<br>' +
                                list(map(lambda p: str(p), list(map(lambda x: 'Price/Earnings ratio: {}'.format(x), stocks_tp['pe_ratio'])))) + '<br>' +
                                'Monthly call: ' + stocks_tp['summary_monthly'] + '<br>' +
                                'Weekly call: ' + stocks_tp['summary_weekly'] + '<br>' +
                                'Daily call: ' + stocks_tp['summary_daily']
                                )],
    "layout": Layout(
                title="Malaysia Bursa Analysis: Reward vs Revenue growth(last year), " + end_date.strftime("%m/%d/%Y"),
                xaxis=dict(title='Revenue growth over latest year',
                           titlefont=dict(family='Courier New, monospace',size=16,color='black')),
                yaxis=dict(title='Reward/Growth (Mean Log Returns)',
                           titlefont=dict(family='Courier New, monospace',size=16,color='black'))
                    )
})    

                

####------------------------------ 4. Correlation analysis(Inspect linear relationship) ------------------------------####
stocks_tp_reduced = stocks_tp.loc[:,['name','log_returns']]
stocks_tp_reduced = stocks_tp_reduced.reset_index(drop=True)

stocks_tp_unnest = pd.DataFrame()
for i in range(0,len(stocks_tp_reduced)):
    stocks_tp_reduced['log_returns'][i] = (stocks_tp_reduced['log_returns'][i].assign(name = stocks_tp_reduced['name'][i]).set_index('name').reset_index())
    stocks_tp_unnest = stocks_tp_unnest.append(stocks_tp_reduced['log_returns'][i])                                   

stocks_tp_unnest['Date'] = stocks_tp_unnest['Date'].apply(lambda x: datetime.strptime(x.lower(),("%b %d, %Y")).strftime("%Y-%m-%d"))
stocks_tp_spread = stocks_tp_unnest.pivot_table(values="return_value", index='Date', columns='name').dropna()    

stocks_tp_spread.head(5)
stocks_tp_corr = stocks_tp_spread.corr()
stocks_tp_corr.head(5)
corrplot = sns.clustermap(data=stocks_tp_corr,  metric="correlation", annot=False, cmap='vlag', figsize=(13, 13))



####------------------------------ 5. Time series analysis on chosen pair of potential stocks ------------------------------####
# find out the highest correlation of each stock
stocks_tp_corr.apply(lambda x: abs(x).nlargest(2)[1])
# Time-series analysis (Stocks can be chosen from potential stocks obtained above which are highly correlated)
stock_picked1 = 'Yinson Holdings Bhd'
stock_picked2 = 'Eco World Develop Group'

stock_price1 = stocks_tp.ix[list(stocks_tp['name']).index(stock_picked1),'historical_price'].copy()          
stock_price2 = stocks_tp.ix[list(stocks_tp['name']).index(stock_picked2),'historical_price'].copy()  

stock_price1['Date'] = pd.to_datetime(stock_price1['Date'].apply(lambda x: datetime.strptime(x.lower(),("%b %d, %Y")).strftime("%Y-%m-%d")))
stock_price2['Date'] = pd.to_datetime(stock_price2['Date'].apply(lambda x: datetime.strptime(x.lower(),("%b %d, %Y")).strftime("%Y-%m-%d")))

stock_price1 = stock_price1.sort_values('Date').reset_index(drop=True)
stock_price2 = stock_price2.sort_values('Date').reset_index(drop=True)

# Growth over last 5 years
start1 = stock_price1['y'].iloc[0]
end1 = stock_price1['y'].iloc[-1]
start2 = stock_price2['y'].iloc[0]
end2 = stock_price2['y'].iloc[-1]

growth_past_stock1 = (stock_price1['y'].iloc[-1] - stock_price1['y'].iloc[0]) / stock_price1['y'].iloc[0]
growth_past_stock2 = (stock_price2['y'].iloc[-1] - stock_price2['y'].iloc[0]) / stock_price2['y'].iloc[0]

day1 = (stock_price1['ds'].iloc[-1] - stock_price1['ds'].iloc[0]).days
day2 = (stock_price2['ds'].iloc[-1] - stock_price2['ds'].iloc[0]).days

CAGR1 = ((stock_price1['y'].iloc[-1] / stock_price1['y'].iloc[0]) **(1/(day1/tradeday_per_year))) - 1
CAGR2 = ((stock_price2['y'].iloc[-1] / stock_price2['y'].iloc[0]) **(1/(day2/tradeday_per_year))) - 1

print('Based on our calculation, the 5 year growth of stock price of \n' + 
      stock_picked1 + ' is ' + str(round(growth_past_stock1*100,2)) + ' % from RM' + str(start1) + ' to RM'+ str(end1) + ',\n' +
      stock_picked2 + ' is ' + str(round(growth_past_stock2*100,2)) +' % from RM' + str(start2) + ' to RM'+ str(end2) + '.')

print('CAGR over last 5 years of \n'+
     stock_picked1 + ' is ' + str(round(CAGR1*100,2)) + ' %,\n' +
     stock_picked2 + ' is ' + str(round(CAGR2*100,2)) + '%.' )

# Time series modeling
# Model time series with Prophet
col_keep = ['Date', 'Price']
stock_price1 = stock_price1[col_keep]
stock_price2 = stock_price2[col_keep]
# Prophet requires columns ds (Date) and y (value)
stock_price1 = stock_price1.rename(columns={'Date': 'ds','Price': 'y'})
stock_price2 = stock_price2.rename(columns={'Date': 'ds','Price': 'y'})

# Make the prophet model and fit on the stock prices of Yinson
stock1_prophet = Prophet(interval_width=0.95, changepoint_prior_scale=0.05, daily_seasonality=False)
stock1_prophet.fit(stock_price1)

# Make a future dataframe for 1 year
stock1_forecast = stock1_prophet.make_future_dataframe(periods=365 * 1, freq='D')
# Make predictions
stock1_forecast = stock1_prophet.predict(stock1_forecast)
stock1_forecast.tail()
stock1_prophet.changepoints

stock1_prophet.plot(stock1_forecast, xlabel = 'Date', ylabel = 'Price (RM)')
plt.title('Stock Price of '+stock_picked1, size=18)
stock1_prophet.plot_components(stock1_forecast)

# Make the prophet model and fit on the stocks prices of Eco World
stock2_prophet = Prophet(interval_width=0.95, changepoint_prior_scale=0.05, daily_seasonality=False)
stock2_prophet.fit(stock_price2)

# Make a future dataframe for 1 year
stock2_forecast = stock2_prophet.make_future_dataframe(periods=365 * 1, freq='D')
# Make predictions
stock2_forecast = stock2_prophet.predict(stock2_forecast)
stock2_forecast.tail()
stock2_prophet.changepoints

stock2_prophet.plot(stock2_forecast, xlabel = 'Date', ylabel = 'Price (RM)')
plt.title('Stock Price of '+stock_picked2, size=18)
stock2_prophet.plot_components(stock2_forecast)


# Cross Validation(Model Evaluation)
# cross-validation
stock1_cv = cross_validation(stock1_prophet, initial = '1095 days', horizon = '365 days', period = '45 days')
stock2_cv = cross_validation(stock2_prophet,  initial = '1095 days', horizon = '365 days', period = '45 days')

stock1_cv.head()
stock2_cv.head()

metrics_calculation(data = stock1_cv)
metrics_calculation(data = stock2_cv)

# Comparison of two stocks
# Combine the trends of stocks picked
stock1_names = [stock_picked1+'_%s' % column for column in stock1_forecast.columns]
stock2_names = [stock_picked2+'_%s' % column for column in stock2_forecast.columns]

# merge the dataframe
copy_stock1_forecast = stock1_forecast.copy()
copy_stock2_forecast = stock2_forecast.copy()

# Rename the columns
copy_stock1_forecast.columns = stock1_names
copy_stock2_forecast.columns = stock2_names

# Merge two datasets
forecast = pd.merge(copy_stock1_forecast, copy_stock2_forecast, how = 'inner',
                    left_on = stock_picked1+'_ds', right_on = stock_picked2+'_ds')

# make sure the merging is successful
forecast.isnull().any()
# Rename date column
forecast = forecast.rename(columns={stock_picked1+'_ds': 'Date'}).drop(stock_picked2+'_ds',axis=1)

# Create subplots to set figure size
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8));
# Plot estimate and uncertainty for stock1
ax2.plot(forecast['Date'], forecast[stock_picked1+'_yhat'], label = stock_picked1+' prediction');
ax2.fill_between(forecast['Date'].dt.to_pydatetime(), forecast[stock_picked1+'_yhat_upper'], forecast[stock_picked1+'_yhat_lower'], alpha=0.6, edgecolor = 'k');
# Plot estimate and uncertainty for stock2
ax2.plot(forecast['Date'], forecast[stock_picked2+'_yhat'], 'r', label = stock_picked2+' prediction');
ax2.fill_between(forecast['Date'].dt.to_pydatetime(), forecast[stock_picked2+'_yhat_upper'], forecast[stock_picked2+'_yhat_lower'], alpha=0.6, edgecolor = 'k');
plt.legend();
plt.xlabel('Date', size=14); plt.ylabel('Price (RM)', size=14); plt.title('Stock Price Prediction for '+stock_picked1+' and '+stock_picked2, size=20);

# Expected value in next 1 year
# Extract the expected value in 1 years later 
value_end_stock1 = stock1_forecast['yhat'].iloc[-1]
value_end_stock2 = stock2_forecast['yhat'].iloc[-1]
# Extract the last day in our forecasting period
day_end_stock1 = stock1_forecast['ds'].iloc[-1]
day_end_stock2 = stock2_forecast['ds'].iloc[-1]

# Growth
growth_stock1 = (value_end_stock1 - stock_price1['y'].iloc[-1]) / stock_price1['y'].iloc[-1]
growth_stock2 = (value_end_stock2 - stock_price2['y'].iloc[-1]) / stock_price2['y'].iloc[-1]

print('Based on our analysis, we expect the stock price of \n' + 
      stock_picked1 + ' is RM' + str(round(value_end_stock1,3)) + ' on ' + str(day_end_stock1) + ' , change from last price is '+ str(round(growth_stock1*100,2)) + '%,\n' +
      stock_picked2 + ' is RM' + str(round(value_end_stock2,3)) +' on ' + str(day_end_stock2) + ' , change from last price is '+ str(round(growth_stock2*100,2)) + '%.')


















### Time-series analysis (Stocks can be chosen from potential stocks obtained above which are high correlations)
plt.style.use('fivethirtyeight')
stock_picked1 = 'Yinson Holdings Bhd'
#stock_picked2 = 'Unisem M Bhd'
stock_picked2 = 'Eco World Develop Group'

stock_price1 = stocks_tp.ix[list(stocks_tp['name']).index(stock_picked1),'historical_price'].copy()          
stock_price2 = stocks_tp.ix[list(stocks_tp['name']).index(stock_picked2),'historical_price'].copy()  

stock_price1['Date'] = pd.to_datetime(stock_price1['Date'].apply(lambda x: datetime.strptime(x.lower(),("%b %d, %Y")).strftime("%Y-%m-%d")))
stock_price2['Date'] = pd.to_datetime(stock_price2['Date'].apply(lambda x: datetime.strptime(x.lower(),("%b %d, %Y")).strftime("%Y-%m-%d")))

stock_price1 = stock_price1.sort_values('Date').reset_index(drop=True)
stock_price2 = stock_price2.sort_values('Date').reset_index(drop=True)

# Model time series with Prophet
col_keep = ['Date', 'Price']
stock_price1 = stock_price1[col_keep]
stock_price2 = stock_price2[col_keep]
# Prophet requires columns ds (Date) and y (value)
stock_price1 = stock_price1.rename(columns={'Date': 'ds','Price': 'y'})
stock_price2 = stock_price2.rename(columns={'Date': 'ds','Price': 'y'})

# Make the prophet model and fit on the data
stock1_prophet = Prophet(interval_width=0.95, changepoint_prior_scale=0.05, daily_seasonality=False)
stock1_prophet.fit(stock_price1)

# Make a future dataframe for 1 year
stock1_forecast = stock1_prophet.make_future_dataframe(periods=365 * 1, freq='D')
# Make predictions
stock1_forecast = stock1_prophet.predict(stock1_forecast)
stock1_prophet.changepoints
stock1_forecast.tail()

stock1_prophet.plot(stock1_forecast, xlabel = 'Date', ylabel = 'Price (RM)')
plt.title('Stock Price of '+stock_picked1, size=18)
stock1_prophet.plot_components(stock1_forecast)


#
stock2_prophet = Prophet(interval_width=0.95, changepoint_prior_scale=0.05, daily_seasonality=False)
stock2_prophet.fit(stock_price2)

# Make a future dataframe for 1 year
stock2_forecast = stock2_prophet.make_future_dataframe(periods=365 * 1, freq='D')
# Make predictions
stock2_forecast = stock2_prophet.predict(stock2_forecast)
stock2_prophet.changepoints
stock2_forecast.tail()

stock2_prophet.plot(stock2_forecast, xlabel = 'Date', ylabel = 'Price (RM)')
plt.title('Stock Price of '+stock_picked2, size=18)
stock2_prophet.plot_components(stock2_forecast)


# Combine the trends of stocks picked
stock1_names = [stock_picked1+'_%s' % column for column in stock1_forecast.columns]
stock2_names = [stock_picked2+'_%s' % column for column in stock2_forecast.columns]

# merge the dataframe
copy_stock1_forecast = stock1_forecast.copy()
copy_stock2_forecast = stock2_forecast.copy()

# Rename the columns
copy_stock1_forecast.columns = stock1_names
copy_stock2_forecast.columns = stock2_names

# Merge two datasets
forecast = pd.merge(copy_stock1_forecast, copy_stock2_forecast, how = 'inner',
                    left_on = stock_picked1+'_ds', right_on = stock_picked2+'_ds')

# make sure the merging is  successful
forecast.isnull().any()
# Rename date column
forecast = forecast.rename(columns={stock_picked1+'_ds': 'Date'}).drop(stock_picked2+'_ds',axis=1)

# 
plt.figure(figsize=(10, 8))
plt.plot(forecast.Date, forecast[stock_picked1+'_yhat'], 'b-', label = stock_picked1+' prediction')
plt.plot(forecast.Date, forecast[stock_picked2+'_yhat'], 'r-', label = stock_picked2+' prediction')
plt.xlabel('Date', size=14); plt.ylabel('Price (RM)', size=14); 
plt.title('Stock Price Prediction for '+stock_picked1+' and '+stock_picked2, size=20)
plt.legend()

# Create subplots to set figure size
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8));
# Plot estimate and uncertainty for stock1
ax2.plot(forecast['Date'], forecast[stock_picked1+'_yhat'], label = stock_picked1+' prediction');
ax2.fill_between(forecast['Date'].dt.to_pydatetime(), forecast[stock_picked1+'_yhat_upper'], forecast[stock_picked1+'_yhat_lower'], alpha=0.6, edgecolor = 'k');
# Plot estimate and uncertainty for stock2
ax2.plot(forecast['Date'], forecast[stock_picked2+'_yhat'], 'r', label = stock_picked2+' prediction');
ax2.fill_between(forecast['Date'].dt.to_pydatetime(), forecast[stock_picked2+'_yhat_upper'], forecast[stock_picked2+'_yhat_lower'], alpha=0.6, edgecolor = 'k');
plt.legend();
plt.xlabel('Date', size=14); plt.ylabel('Price (RM)', size=14); plt.title('Stock Price Prediction for '+stock_picked1+' and '+stock_picked2, size=20);


# Extract the expected value in 1 years later 
value_end_stock1 = stock1_forecast['yhat'].iloc[-1]
value_end_stock2 = stock2_forecast['yhat'].iloc[-1]
# Growth
growth_stock1 = (value_end_stock1 - stock_price1['y'].iloc[-1]) / stock_price1['y'].iloc[-1]
growth_stock2 = (value_end_stock2 - stock_price2['y'].iloc[-1]) / stock_price2['y'].iloc[-1]

print('From our analysis, we expect the stock price of \n' + 
      stock_picked1 + 'is RM' + str(round(value_end_stock1,3)) + ' , changing from last price is '+ str(round(growth_stock1*100,2)) + '%,\n' +
      stock_picked2 + 'is RM' + str(round(value_end_stock2,3)) + ' , changing from last price is '+ str(round(growth_stock2*100,2)) + '%')


# cross-validation
stock1_cv = cross_validation(stock1_prophet, initial = '1095 days', horizon = '365 days', period = '45 days')
stock1_cv.head()
stock2_cv = cross_validation(stock2_prophet,  initial = '1095 days', horizon = '365 days', period = '45 days')
stock2_cv.head()

metrics_stock1 = performance_metrics(stock1_cv)
metrics_stock2 = performance_metrics(stock2_cv)

stock1_cv['day_diff'] = (stock1_cv['ds'] - stock1_cv['cutoff'])/ np.timedelta64(1, 'D')
stock1_cv['ape'] = abs(stock1_cv['y'] - stock1_cv['yhat'])/stock1_cv['y']

stock2_cv['day_diff'] = (stock2_cv['ds'] - stock2_cv['cutoff'])/ np.timedelta64(1, 'D')
stock2_cv['ape'] = abs(stock2_cv['y'] - stock2_cv['yhat'])/stock2_cv['y']

plt.scatter(x=stock1_cv['day_diff'], y=stock1_cv['ape'])
plt.scatter(x=stock2_cv['day_diff'], y=stock2_cv['ape'])

    
    
    
    
    