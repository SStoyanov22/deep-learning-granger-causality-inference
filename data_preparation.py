# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:17:51 2022

@author: stoyan.stoyanov
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:03:45 2022

@author: stoyan.stoyanov
"""
#%%
import pandas as pd
from yahoofinancials import YahooFinancials
from functools import reduce

class historical_data_collector:
    def __init__(self,start_date,end_date,tickers):
            self.start_date = start_date
            self.end_date = end_date
            self.tickers = tickers
            self.tickers_data = self.get_all_tickers_data()
            self.df = self.get_df()

    def get_all_tickers_data(self):
      #  t_data = [(lambda name, ticker:self.get_ticker_data(name,ticker)) for name,ticker in self.tickers.items()]
        t_data = []
        
        for name,ticker in self.tickers.items():
            df = self.get_ticker_data(name, ticker)
            t_data.append(df)
        return t_data

    def get_ticker_data(self, name,ticker):
        yf = YahooFinancials(ticker)
        data=yf.get_historical_price_data(self.start_date, self.end_date, "daily")
        df = pd.DataFrame(data[ticker]['prices']) 
        
        #rename close column to ticker name
        df = df.rename(columns = {"close":name})
        
        #Put all unecessarry columns in an array
        to_drop = ['high', 'low', 'open', 'volume', 'adjclose', 
               'formatted_date']

        #drop unecessary columns, create date column and set as index
        df.date = pd.to_datetime(df.formatted_date)
        df = df.drop(to_drop, axis=1).set_index("date")

        #Remove duplicated indexes/dates for BTC
        if df.index.duplicated().any():
            df = df.loc[~df.index.duplicated(), :]
        
        return df
    
    def get_df(self):
        merged = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, 
                                              left_index=True, right_index=True, 
                                              how='outer'), self.tickers_data)
        merged.fillna(method='ffill',inplace = True)
        merged['Prediction'] = merged['BTC']
        return merged
