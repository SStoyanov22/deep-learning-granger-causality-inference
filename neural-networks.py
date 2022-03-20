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
START_DATE = "2014-09-17"
END_DATE = "2022-03-20"

pd.set_option('display.max_columns', 7)
"""Get Data"""


"""BTC"""
print("BTC")
btc_financials = YahooFinancials(TICKER_BTC)
data=btc_financials.get_historical_price_data(START_DATE, END_DATE, "daily")
btc_df = pd.DataFrame(data[TICKER_BTC]['prices'])
btc_df = btc_df.drop('date', axis=1).set_index('formatted_date')
#print(btc_df.head())

"""NASDAQ"""
print("NASDAQ")
nasdaq_financials = YahooFinancials(TICKER_NASDAQ)
data=nasdaq_financials.get_historical_price_data(START_DATE, END_DATE, "daily")
nasdaq_df = pd.DataFrame(data[TICKER_NASDAQ]['prices'])
nasdaq_df = nasdaq_df.drop('date', axis=1).set_index('formatted_date')
#print(nasdaq_df.head())

"""S&P 500"""
print("SNP")
snp_financials = YahooFinancials(TICKER_SNP)
data=snp_financials.get_historical_price_data(START_DATE, END_DATE, "daily")
snp_df = pd.DataFrame(data[TICKER_SNP]['prices'])
snp_df = snp_df.drop('date', axis=1).set_index('formatted_date')
#print(snp_df.head())


"""Drop unecessarry columns"""
to_drop = ["high","low","open","adjclose","volume"]
btc_df=  btc_df.drop(to_drop, axis=1)
nasdaq_df=  nasdaq_df.drop(to_drop, axis=1)
snp_df=  snp_df.drop(to_drop, axis=1)
print(btc_df)