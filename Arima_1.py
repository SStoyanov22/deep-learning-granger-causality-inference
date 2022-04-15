#import libraties

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob 
from statsmodels import api as sm
from scipy.stats import linregress
import matplotlib.pyplot as plt
#conda install yahoofinancials
from yahoofinancials import YahooFinancials


"""Constants"""
TICKER_BTC = "BTC-USD"
TICKER_NASDAQ = "^IXIC"
TICKER_SNP = "^GSPC"
START_DATE = "2014-09-17"
END_DATE = "2022-04-07"
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 100)

# ### Get data for BTC

btc_financials = YahooFinancials(TICKER_BTC)
data=btc_financials.get_historical_price_data(START_DATE, END_DATE, "daily")
btc_df = pd.DataFrame(data[TICKER_BTC]['prices'])


# ### Get data for S&P 500


snp_financials = YahooFinancials(TICKER_SNP)
data=snp_financials.get_historical_price_data(START_DATE, END_DATE, "daily")
snp_df = pd.DataFrame(data[TICKER_SNP]['prices'])


# ## Data preparation

#Put all unecessarry columns in an array
to_drop = ['high', 'low', 'open', 'volume', 'adjclose', 
       'formatted_date']


# ### Prepare BTC data

#rename close column to price
btc_df = btc_df.rename(columns = {"close":"BTC"})

#check for date duplicates
len(btc_df['formatted_date'].unique()) #2759
len(btc_df.index) #2759

#drop unecessary columns, create date column and set as index
btc_df.date = pd.to_datetime(btc_df.formatted_date)
btc_df = btc_df.drop(to_drop, axis=1).set_index("date")

#Remove duplicated indexes/dates for BTC
if btc_df.index.duplicated().any():
    btc_df = btc_df.loc[~btc_df.index.duplicated(), :]

# ### Prepare S&P 500 data

#rename close column to price
snp_df = snp_df.rename(columns = {"close":"SP500"})

#check for date duplicates
len(snp_df['formatted_date'].unique()) #1902
len(snp_df.index) #1902

#drop unecessary columns, create date column and set as index
snp_df.date = pd.to_datetime(snp_df.formatted_date)
snp_df = snp_df.drop(to_drop, axis=1).set_index("date")

#Remove duplicated indexes/dates for BTC
if snp_df.index.duplicated().any():
    snp_df = snp_df.loc[~snp_df.index.duplicated(), :]

snp_df.head(10)


# ### Merge both dataframes into one

#Data frame with NaN values that is going to be used for imputation and modelling
df = btc_df.merge(snp_df, how="outer", left_index=True, right_index=True)

#Reindex to get all dates
df.reindex(pd.date_range(start=min(df.index), end=max(df.index), freq='D'))

#using forward propagation to fill all missing date values for SNP500
df.fillna(method='ffill', inplace=True)

###############################ARIMA MODEL###############################

#check data for stationarity
from statsmodels.tsa.stattools import adfuller
plt.plot('BTC',data=df)
adfuller(df["BTC"]) # p value 0.76

#value is not stationary and has to be normalized


from statsmodels.graphics.tsaplots import plot_acf

#1st order differencing
diff1_plot = plt.figure().add_subplot()
diff1_plot.set_title('1st Order Differencing')
diff1_plot.plot(df['BTC'].diff())

adfuller(df['BTC'].diff().dropna())
#p value < 0.05 therefore data is normalized after 1 period

#acf after 1 differencing
plot_acf(df['BTC'].diff().dropna())

#pacf after 1 differencing
sm.graphics.tsa.plot_pacf(df["BTC"].diff().dropna())


#pip install pmdarima
from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")
#Calling our model and generating best possible ARIMA combination
 #calling our function
auto_model1 = auto_arima(df["BTC"],suppress_warnings=True)           
auto_model1.summary() # model to be 0,1,0

# Create Training and Test
train = df.BTC[:2070]
test = df.BTC[2070:]

# Build Model

from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)
 
model = ARIMA(train, order=(0, 1, 0))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(690, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
