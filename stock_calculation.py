# -*- coding: utf-8 -*-
"""
Created in 2018

@author: JiaRong
"""

import numpy as np
import sys
from sklearn.metrics import mean_absolute_error  
from sklearn.metrics import mean_squared_error  

# Calculate the return of stocks price
def get_log_returns(hist_data, period_options):
    
    '''
    df['pct_change'] = df.price.pct_change()
    df['log_return'] = np.log(1 + df.pct_change)
    '''
    hist_data = hist_data.loc[:,['Date','Price']]
    
    if period_options == 'daily':
        hist_data = hist_data.assign(return_value = np.log(1 + hist_data['Price'].pct_change(-1)))
    elif period_options == 'monthly':
        hist_data = hist_data.assign(return_value = np.log(1 + hist_data['Price'].pct_change(-21)))
    elif period_options == 'yearly':
        hist_data = hist_data.assign(return_value = np.log(1 + hist_data['Price'].pct_change(-252)))
    else:
        sys.exit('Please input correct period. The period must be one of "daily","monthly" and "yearly".')
        
    hist_data = hist_data.drop(columns='Price')
    return (hist_data)


# Calculate metrics for time-series model evaluation (MAE, RMSE, MAPE)
def metrics_calculation(data):
    mae = mean_absolute_error(data['y'], data['yhat'])
    rmse = np.sqrt(mean_squared_error(data['y'], data['yhat']))
    mape = (abs(data['y'] - data['yhat'])/data['y']).mean()
    metrics = {'MAE':mae, 'RMSE':rmse, 'MAPE':mape}
    return metrics