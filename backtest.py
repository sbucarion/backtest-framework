import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si
from yahoo_fin.stock_info import *
from yahoo_fin import *
import math
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime as dt
from datetime import timedelta
import datetime
from datetime import date
import time
import singlestock_functions as ssf
import warnings

def stationarity(training_df, trading_df, alpha):
    for ticker in training_df:
        adf = adfuller(training_df[ticker])[1]
        
        if adf > alpha:
            training_df.pop(ticker)
            trading_df.pop(ticker) 
            
def halflife(spread):
    """Regression on the pairs spread to find lookback
        period for trading"""
    x_lag = np.roll(spread,1)
    x_lag[0] = 0
    y_ret = spread - x_lag
    y_ret[0] = 0
    
    x_lag_constant = sm.add_constant(x_lag)
    
    res = sm.OLS(y_ret,x_lag_constant).fit()
    halflife = -np.log(2) / res.params[1]
    halflife = int(round(halflife))
    return halflife 
    
tickers = si.tickers_sp500()

#Set backtest length
time_length = 365 * 5
today = date.today() + timedelta(days = 1)

#Start date for backtest
backtest_start = today - timedelta(days = time_length)

#Get all days market is open
market_dates = get_data("AAPL", start_date = backtest_start).index

#Set alpha bounds
alpha = 0.01



for ticker in tickers:
    #Input all data for backtest into dataframe
    data_df = pd.DataFrame(get_data(ticker, 
                                    start_date = market_dates[0] - timedelta(days = 67, weeks = 260), 
                                    end_date = today))
    
    data_df = data_df['adjclose'] #Select only closed data
    
    
    #Simulate dates to mimic current trading day
    for date in market_dates:
        ed = date
        sd = ed - timedelta(days = 67, weeks = 260)

        #Use only 5 year slice of data for trading info
        current_data = data_df.loc[sd:ed]
        
        if len(current_data) < 1240:
            continue

            
        try:
            #Split data into training and testing
            coint_df = current_data[:-100]
            trading_df = current_data[-100:]
            
            #Check data for stationarity
            if adfuller(coint_df)[1] < alpha:
                hl = halflife(coint_df)

                spread = trading_df[-hl:]
                spread = (spread - spread.mean()) / np.std(spread)

                #Signal Long or Short Trades for cureent day
                if spread[-1] > 2 and spread[-2] < 2:
                    print("close", ticker, trading_df[date], date)
                    

                if spread[-1] < -2 and spread[-2] > -2:
                    print('long', ticker, trading_df[date], date)
                    
                    
        except:
            continue
