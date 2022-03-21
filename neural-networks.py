# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:44:29 2022

@authors:   vladimir.milchev
            kristofar.stavrev
            radoslava.dencheva
            stoyan.stoyanov
          
"""
"""Import all necessary libraries"""
import pandas as pd
import numpy as np
import glob 
#!pip install yfinance
#!pip install yahoofinancials
import yfinance as yf
from yahoofinancials import YahooFinancials

"""Constants"""
TICKER_BTC = "BTC-USD"
TICKER_NASDAQ = "^IXIC"
TICKER_SNP = "^GSPC"
START_DATE = "2014-01-01"
END_DATE = "2022-03-20"

pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 100)
"""Get Data"""

# """BTC"""
print("BTC")
btc_financials = YahooFinancials(TICKER_BTC)
data=btc_financials.get_historical_price_data(START_DATE, END_DATE, "daily")
btc_df = pd.DataFrame(data[TICKER_BTC]['prices'])

"""S&P 500"""
print("S&P 500")
snp_financials = YahooFinancials(TICKER_SNP)
data=snp_financials.get_historical_price_data(START_DATE, END_DATE, "daily")
snp_df = pd.DataFrame(data[TICKER_SNP]['prices'])


"""Drop unnecessary columns and rename others"""
to_drop = ['high', 'low', 'open', 'volume', 'adjclose', 
       'formatted_date']
snp_df.date = pd.to_datetime(snp_df.formatted_date)
snp_df = snp_df.rename(columns = {"close":"price"})
snp_df=  snp_df.drop(to_drop, axis=1).set_index('date')

btc_df.date = pd.to_datetime(btc_df.formatted_date)
btc_df = btc_df.rename(columns = {"close":"price"})
btc_df=  btc_df.drop(to_drop, axis=1).set_index('date')

"""Set 'date' as index"""

#Remove duplicated indexes/dates for S&P 500
if snp_df.index.duplicated().any():
    snp_df = snp_df.loc[~snp_df.index.duplicated(), :]
snp_df= snp_df.reindex(pd.date_range(start=min(snp_df.index), end=max(snp_df.index), freq='D'))
snp_df.index.name = 'date'

#Remove duplicated indexes/dates for BTC
if btc_df.index.duplicated().any():
    btc_df = btc_df.loc[~btc_df.index.duplicated(), :]
btc_df= btc_df.reindex(pd.date_range(start=min(btc_df.index), end=max(btc_df.index), freq='D'))
btc_df.index.name = 'date'

print(len(btc_df.index))
print(len(snp_df.index))

print(btc_df.price.isnull().sum())
print(snp_df.price.isnull().sum())

